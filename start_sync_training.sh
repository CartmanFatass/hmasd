#!/bin/bash

# HMASD同步训练启动脚本

set -e  # 出错时退出

echo "=================================================="
echo "        HMASD同步训练启动脚本"
echo "=================================================="

# 检查Python环境
echo "检查Python环境..."
if ! command -v python &> /dev/null; then
    echo "错误：未找到Python"
    exit 1
fi

# 检查必要的包
echo "检查依赖包..."
python -c "import torch, numpy, tensorboardX" 2>/dev/null || {
    echo "错误：缺少必要的Python包"
    echo "请安装：pip install torch numpy tensorboardX"
    exit 1
}

# 创建必要的目录
echo "创建目录..."
mkdir -p models
mkdir -p tf-logs
mkdir -p logs

# 设置环境变量
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export CUDA_VISIBLE_DEVICES=0

# 解析命令行参数
MODE="train"
LOG_LEVEL="INFO"
MODEL_PATH=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --eval-only)
            MODE="eval"
            shift
            ;;
        --model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --log-level)
            LOG_LEVEL="$2"
            shift 2
            ;;
        --help|-h)
            echo "用法: $0 [选项]"
            echo "选项:"
            echo "  --eval-only           仅运行评估模式"
            echo "  --model-path PATH     指定模型路径"
            echo "  --log-level LEVEL     日志级别 (DEBUG/INFO/WARNING/ERROR)"
            echo "  --help, -h            显示此帮助信息"
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            echo "使用 --help 查看帮助信息"
            exit 1
            ;;
    esac
done

# 构建命令
CMD="python train_sync_enhanced.py --log-level $LOG_LEVEL"

if [ "$MODE" = "eval" ]; then
    CMD="$CMD --eval-only"
    if [ -z "$MODEL_PATH" ]; then
        echo "错误：评估模式需要指定模型路径"
        echo "使用: $0 --eval-only --model-path models/your_model.pth"
        exit 1
    fi
fi

if [ -n "$MODEL_PATH" ]; then
    CMD="$CMD --model-path $MODEL_PATH"
fi

# 显示配置信息
echo ""
echo "配置信息:"
echo "  模式: $MODE"
echo "  日志级别: $LOG_LEVEL"
echo "  模型路径: ${MODEL_PATH:-"无"}"
echo "  GPU设备: ${CUDA_VISIBLE_DEVICES:-"自动"}"
echo ""

# 检查GPU
if command -v nvidia-smi &> /dev/null; then
    echo "GPU状态:"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
    echo ""
fi

# 启动TensorBoard（后台运行）
if [ "$MODE" = "train" ]; then
    echo "启动TensorBoard..."
    pkill -f "tensorboard.*tf-logs" 2>/dev/null || true
    nohup tensorboard --logdir tf-logs --port 6006 --bind_all > /dev/null 2>&1 &
    TENSORBOARD_PID=$!
    echo "TensorBoard已启动，访问地址: http://localhost:6006"
    echo "TensorBoard PID: $TENSORBOARD_PID"
    echo ""
fi

# 显示训练命令
echo "执行命令: $CMD"
echo ""

# 开始训练/评估
if [ "$MODE" = "train" ]; then
    echo "开始同步训练..."
    echo "提示：可以随时按 Ctrl+C 停止训练"
    echo ""
else
    echo "开始评估..."
    echo ""
fi

# 捕获中断信号
trap 'echo ""; echo "收到中断信号，正在清理..."; pkill -P $$ 2>/dev/null || true; exit 1' INT TERM

# 执行主命令
eval $CMD

# 训练/评估完成
if [ "$MODE" = "train" ]; then
    echo ""
    echo "训练完成！"
    echo "模型保存在: models/"
    echo "日志保存在: tf-logs/"
    echo ""
    echo "查看训练结果:"
    echo "  TensorBoard: http://localhost:6006"
    echo "  模型文件: ls -la models/"
    echo "  日志文件: ls -la logs/"
else
    echo ""
    echo "评估完成！"
fi

echo ""
echo "=================================================="
echo "        脚本执行完成"
echo "=================================================="
