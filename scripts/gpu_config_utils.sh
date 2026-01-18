#!/bin/bash
# Auto-detect PyTorch flavour and set optimal environment variables
# Returns: cuda, rocm, or cpu - matching pyproject.toml optional-dependencies
# Note: 'gpu' is an alias for 'cuda' (backward compatibility)

detect_torch_flavour() {
    echo "Auto-detecting PyTorch flavour..." >&2
    
    # Check for ROCm first (AMD GPUs)
    if command -v rocminfo &> /dev/null && rocminfo &> /dev/null 2>&1; then
        echo "  → Detected ROCm (AMD GPU via rocminfo)" >&2
        echo "rocm"
        return 0
    fi
    
    # Check for CUDA (NVIDIA GPUs)
    if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null 2>&1; then
        echo "  → Detected CUDA (NVIDIA GPU via nvidia-smi)" >&2
        echo "cuda"
        return 0
    fi
    
    # Fallback to CPU
    echo "  → No GPU detected, using CPU backend" >&2
    echo "cpu"
    return 0
}

detect_num_gpus() {
    local torch_flavour=$1
    
    # Normalize 'gpu' to 'cuda' (gpu is an alias)
    if [ "$torch_flavour" = "gpu" ]; then
        torch_flavour="cuda"
    fi
    
    if [ "$torch_flavour" = "rocm" ]; then
        # Count ROCm GPU devices (exclude CPU devices)
        # Try using rocminfo first, but fall back to Python/PyTorch if available
        if command -v rocminfo &> /dev/null; then
            # Count GPU devices: look for Device Type lines containing "GPU" but not "CPU"
            # rocminfo may show both CPU and GPU devices, we only want GPUs
            local count=$(rocminfo 2>/dev/null | awk '
                /^Device Type:/ {
                    line = $0
                    # Check if this line contains GPU but not CPU
                    if (line ~ /GPU/ && line !~ /CPU/) {
                        count++
                    }
                }
                END {
                    print (count ? count : 0)
                }
            ')
            # If rocminfo didn't find GPUs or returned empty, try PyTorch as fallback
            if [ -z "$count" ] || [ "$count" = "0" ]; then
                python3 -c "import torch; print(torch.cuda.device_count() if torch.cuda.is_available() and hasattr(torch.version, 'hip') and torch.version.hip else 0)" 2>/dev/null || echo "0"
            else
                echo "$count"
            fi
        else
            # Fallback: try Python/PyTorch if available
            python3 -c "import torch; print(torch.cuda.device_count() if torch.cuda.is_available() and hasattr(torch.version, 'hip') and torch.version.hip else 0)" 2>/dev/null || echo "0"
        fi
    elif [ "$torch_flavour" = "cuda" ]; then
        # Count CUDA devices
        if command -v nvidia-smi &> /dev/null; then
            nvidia-smi --list-gpus 2>/dev/null | wc -l || echo "0"
        else
            echo "0"
        fi
    else
        # CPU: return 1 (single process)
        echo "1"
    fi
}

show_gpu_info() {
    local torch_flavour=$1
    
    # Normalize 'gpu' to 'cuda' (gpu is an alias)
    if [ "$torch_flavour" = "gpu" ]; then
        torch_flavour="cuda"
    fi
    
    # Detect number of GPUs
    local num_gpus=$(detect_num_gpus "$torch_flavour")
    
    echo ""
    echo "PyTorch Backend Information:"
    echo "============================"
    
    if [ "$torch_flavour" = "rocm" ]; then
        if command -v rocminfo &> /dev/null; then
            echo "Backend: ROCm (AMD GPU)"
            echo "  Number of GPUs: $num_gpus"
            # Show GPU information only (filter out CPU devices) if GPUs are present
            if [ "$num_gpus" -gt 0 ]; then
                # Try to extract GPU info from rocminfo
                local gpu_info=$(rocminfo 2>/dev/null | awk '
                    BEGIN { in_gpu = 0 }
                    /^Device Type:/ {
                        if ($0 ~ /GPU/ && $0 !~ /CPU/) {
                            in_gpu = 1
                            print $0
                        } else {
                            in_gpu = 0
                        }
                    }
                    in_gpu && /^  Marketing Name:/ { print $0 }
                ' | head -2)
                if [ -n "$gpu_info" ]; then
                    echo "$gpu_info" | sed 's/^/  /'
                else
                    # Fallback: try PyTorch to get GPU name (if available)
                    python3 -c "import torch; print(f'  GPU: {torch.cuda.get_device_name(0)}') if torch.cuda.is_available() and hasattr(torch.version, 'hip') and torch.version.hip else None" 2>/dev/null || echo "  (rocminfo details unavailable, will verify after PyTorch installation)"
                fi
            else
                echo "  (No GPU devices found via rocminfo, will verify after PyTorch installation)"
            fi
        fi
    elif [ "$torch_flavour" = "cuda" ]; then
        if command -v nvidia-smi &> /dev/null; then
            echo "Backend: CUDA (NVIDIA GPU)"
            echo "  Number of GPUs: $num_gpus"
            nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null | head -1 | \
                awk -F', ' '{printf "  GPU: %s\n  Driver: %s\n  Memory: %s\n", $1, $2, $3}' || echo "  (nvidia-smi details unavailable)"
        fi
    else
        echo "Backend: CPU (no GPU)"
        echo "  CPU cores: $(nproc 2>/dev/null || echo 'unknown')"
        echo "  Processes: $num_gpus"
    fi
    echo ""
}

set_backend_env_vars() {
    local torch_flavour=$1
    
    # Normalize 'gpu' to 'cuda' (gpu is an alias)
    if [ "$torch_flavour" = "gpu" ]; then
        torch_flavour="cuda"
    fi
    
    if [ "$torch_flavour" = "rocm" ]; then
        echo "Setting ROCm-optimized environment variables..."
        export FLASH_ATTENTION_TRITON_AMD_ENABLE="FALSE"
        export USE_ROCM_CK_SDPA=1
        export USE_ROCM_CK_GEMM=1
        export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
        # ROCm-specific optimizations
        export HIP_LAUNCH_BLOCKING=0
    elif [ "$torch_flavour" = "cuda" ]; then
        echo "Setting CUDA-optimized environment variables..."
    else
        echo "Using CPU backend (no GPU-specific optimizations)"
    fi
    
    # Common optimizations for all backends
    export OMP_NUM_THREADS=1
    export TOKENIZERS_PARALLELISM=false
}

verify_backend_after_install() {
    echo "Verifying PyTorch backend after installation..."
    python3 -c "
import torch
if torch.cuda.is_available():
    if hasattr(torch.version, 'hip') and torch.version.hip is not None:
        print(f'✓ ROCm backend active (HIP version: {torch.version.hip})')
    else:
        print(f'✓ CUDA backend active (CUDA version: {torch.version.cuda})')
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  Device count: {torch.cuda.device_count()}')
else:
    print('✓ CPU backend active (no GPU detected)')
" 2>/dev/null || echo "Warning: Could not verify backend"
}

# Main execution if script is run directly
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    TORCH_FLAVOUR=$(detect_torch_flavour)
    show_gpu_info "$TORCH_FLAVOUR"
    set_backend_env_vars "$TORCH_FLAVOUR"
    echo "Detected PyTorch flavour: $TORCH_FLAVOUR"
fi
