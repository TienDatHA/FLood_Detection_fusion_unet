#!/bin/bash

# 🐳 Docker Management Script for Flood Detection System
# Usage: ./docker-helper.sh [command] [options]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Project name
PROJECT_NAME="flood-detection"

# Helper functions
print_usage() {
    echo -e "${BLUE}🐳 Docker Helper Script for Flood Detection System${NC}"
    echo ""
    echo "Usage: $0 [command] [options]"
    echo ""
    echo -e "${YELLOW}Commands:${NC}"
    echo "  build         Build all Docker images"
    echo "  start         Start main flood detection service"
    echo "  dev           Start development environment (Jupyter + TensorBoard)"
    echo "  train         Run training"
    echo "  inference     Run inference"
    echo "  notebook      Open Jupyter notebook"
    echo "  tensorboard   Start TensorBoard"
    echo "  shell         Open bash shell in container"
    echo "  logs          Show logs"
    echo "  status        Show container status"
    echo "  stop          Stop all services"
    echo "  clean         Clean up containers and images"
    echo "  gpu-test      Test GPU availability"
    echo ""
    echo -e "${YELLOW}Examples:${NC}"
    echo "  $0 build          # Build images"
    echo "  $0 dev            # Start development environment"
    echo "  $0 train          # Run training"
    echo "  $0 inference      # Run inference"
    echo "  $0 shell          # Open shell"
    echo "  $0 clean          # Cleanup"
}

check_requirements() {
    echo -e "${BLUE}🔍 Checking requirements...${NC}"
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}❌ Docker not found. Please install Docker first.${NC}"
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        echo -e "${RED}❌ Docker Compose not found. Please install Docker Compose first.${NC}"
        exit 1
    fi
    
    # Check NVIDIA Docker (optional)
    if command -v nvidia-smi &> /dev/null; then
        echo -e "${GREEN}✅ NVIDIA GPU detected${NC}"
        if ! docker run --rm --gpus all nvidia/cuda:11.8-base nvidia-smi &> /dev/null; then
            echo -e "${YELLOW}⚠️  NVIDIA Docker not properly configured. GPU support may not work.${NC}"
        else
            echo -e "${GREEN}✅ NVIDIA Docker working${NC}"
        fi
    else
        echo -e "${YELLOW}⚠️  No NVIDIA GPU detected. Running in CPU mode.${NC}"
    fi
    
    echo -e "${GREEN}✅ Requirements check completed${NC}"
}

setup_directories() {
    echo -e "${BLUE}📁 Setting up directories...${NC}"
    
    # Create necessary directories
    mkdir -p data outputs models logs training_logs evaluation_logs
    mkdir -p inference_results visual_outputs dem_visualizations
    mkdir -p final_flood_results eval_results notebooks
    
    echo -e "${GREEN}✅ Directories created${NC}"
}

build_images() {
    echo -e "${BLUE}🏗️  Building Docker images...${NC}"
    
    docker-compose build
    
    echo -e "${GREEN}✅ Build completed${NC}"
}

start_main() {
    echo -e "${BLUE}🚀 Starting main flood detection service...${NC}"
    
    setup_directories
    docker-compose up -d flood-detection
    
    echo -e "${GREEN}✅ Service started${NC}"
    echo -e "${YELLOW}💡 Use '$0 shell' to access the container${NC}"
}

start_dev() {
    echo -e "${BLUE}🔧 Starting development environment...${NC}"
    
    setup_directories
    docker-compose --profile dev up -d
    
    echo -e "${GREEN}✅ Development environment started${NC}"
    echo -e "${YELLOW}💡 Jupyter Lab: http://localhost:8889${NC}"
    echo -e "${YELLOW}💡 TensorBoard: http://localhost:6007${NC}"
}

run_training() {
    echo -e "${BLUE}🎯 Starting training...${NC}"
    
    setup_directories
    docker-compose --profile training up
    
    echo -e "${GREEN}✅ Training completed${NC}"
}

run_inference() {
    echo -e "${BLUE}🔍 Running inference...${NC}"
    
    setup_directories
    docker-compose --profile inference up
    
    echo -e "${GREEN}✅ Inference completed${NC}"
}

open_notebook() {
    echo -e "${BLUE}📓 Opening Jupyter notebook...${NC}"
    
    if ! docker-compose ps jupyter | grep -q "Up"; then
        echo -e "${YELLOW}⚠️  Starting Jupyter service first...${NC}"
        docker-compose --profile dev up -d jupyter
        sleep 5
    fi
    
    echo -e "${GREEN}✅ Jupyter Lab available at: http://localhost:8889${NC}"
    
    # Try to open in browser (Linux)
    if command -v xdg-open &> /dev/null; then
        xdg-open http://localhost:8889
    elif command -v open &> /dev/null; then
        open http://localhost:8889
    fi
}

start_tensorboard() {
    echo -e "${BLUE}📊 Starting TensorBoard...${NC}"
    
    docker-compose --profile monitoring up -d tensorboard
    
    echo -e "${GREEN}✅ TensorBoard available at: http://localhost:6008${NC}"
    
    # Try to open in browser
    if command -v xdg-open &> /dev/null; then
        xdg-open http://localhost:6008
    elif command -v open &> /dev/null; then
        open http://localhost:6008
    fi
}

open_shell() {
    echo -e "${BLUE}🐚 Opening shell in container...${NC}"
    
    if ! docker-compose ps flood-detection | grep -q "Up"; then
        echo -e "${YELLOW}⚠️  Starting main service first...${NC}"
        start_main
        sleep 3
    fi
    
    docker-compose exec flood-detection bash
}

show_logs() {
    echo -e "${BLUE}📋 Showing logs...${NC}"
    
    docker-compose logs -f --tail=100 flood-detection
}

show_status() {
    echo -e "${BLUE}📊 Container status:${NC}"
    docker-compose ps
    
    echo ""
    echo -e "${BLUE}📊 Resource usage:${NC}"
    docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}"
}

stop_services() {
    echo -e "${BLUE}🛑 Stopping all services...${NC}"
    
    docker-compose down
    
    echo -e "${GREEN}✅ All services stopped${NC}"
}

cleanup() {
    echo -e "${BLUE}🧹 Cleaning up...${NC}"
    
    # Stop all containers
    docker-compose down
    
    # Remove unused containers
    docker container prune -f
    
    # Remove unused images
    docker image prune -f
    
    # Remove unused volumes (careful!)
    echo -e "${YELLOW}⚠️  This will remove unused Docker volumes. Continue? (y/N)${NC}"
    read -r response
    if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        docker volume prune -f
    fi
    
    echo -e "${GREEN}✅ Cleanup completed${NC}"
}

test_gpu() {
    echo -e "${BLUE}🧪 Testing GPU availability...${NC}"
    
    # Test NVIDIA Docker
    if docker run --rm --gpus all nvidia/cuda:11.8-base nvidia-smi; then
        echo -e "${GREEN}✅ GPU test passed${NC}"
    else
        echo -e "${RED}❌ GPU test failed${NC}"
        return 1
    fi
    
    # Test in our container
    if docker-compose ps flood-detection | grep -q "Up"; then
        echo -e "${BLUE}🔍 Testing GPU in flood detection container...${NC}"
        docker-compose exec flood-detection nvidia-smi
    fi
}

# Main script logic
case "${1:-help}" in
    build)
        check_requirements
        build_images
        ;;
    start)
        check_requirements
        start_main
        ;;
    dev)
        check_requirements
        start_dev
        ;;
    train)
        check_requirements
        run_training
        ;;
    inference)
        check_requirements
        run_inference
        ;;
    notebook)
        open_notebook
        ;;
    tensorboard)
        start_tensorboard
        ;;
    shell)
        open_shell
        ;;
    logs)
        show_logs
        ;;
    status)
        show_status
        ;;
    stop)
        stop_services
        ;;
    clean)
        cleanup
        ;;
    gpu-test)
        test_gpu
        ;;
    help|--help|-h)
        print_usage
        ;;
    *)
        echo -e "${RED}❌ Unknown command: $1${NC}"
        print_usage
        exit 1
        ;;
esac