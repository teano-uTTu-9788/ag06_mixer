#!/usr/bin/env bash
# Docker Library - Container management helpers

set -euo pipefail

source "${BASH_SOURCE%/*}/logging.sh"

# Check if Docker is running
docker_running() {
    docker info >/dev/null 2>&1
}

# Wait for Docker to be ready
wait_for_docker() {
    local max_wait="${1:-60}"
    local waited=0
    
    while ! docker_running && [[ $waited -lt $max_wait ]]; do
        log_info "Waiting for Docker to start..."
        sleep 5
        ((waited += 5))
    done
    
    docker_running
}

# Clean up containers and images
docker_cleanup() {
    log_info "Cleaning up Docker resources..."
    
    # Remove stopped containers
    docker container prune -f
    
    # Remove unused images
    docker image prune -f
    
    # Remove unused volumes
    docker volume prune -f
    
    log_success "Docker cleanup complete"
}

# Build with cache
docker_build_cached() {
    local image="$1"
    local context="${2:-.}"
    local dockerfile="${3:-Dockerfile}"
    
    docker build \
        --cache-from "$image:latest" \
        --tag "$image:latest" \
        --file "$dockerfile" \
        "$context"
}

export -f docker_running wait_for_docker docker_cleanup docker_build_cached
