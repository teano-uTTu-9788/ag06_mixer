#!/usr/bin/env bash
# Docker Operations Library - Container Management Pattern
# Provides Docker and container orchestration utilities

# Require core library
deps::require "core"

# ============================================================================
# Docker Installation & Setup (Enterprise Container Pattern)
# ============================================================================

docker::is_installed() {
    command -v docker &>/dev/null
}

docker::is_running() {
    docker info &>/dev/null
}

docker::version() {
    docker --version 2>/dev/null | awk '{print $3}' | tr -d ','
}

docker::compose_version() {
    docker compose version 2>/dev/null | awk '{print $4}'
}

# ============================================================================
# Image Management (Google Container Registry Pattern)
# ============================================================================

docker::build() {
    local dockerfile="${1:-Dockerfile}"
    local tag="${2:-latest}"
    local context="${3:-.}"
    local build_args="${4:-}"
    
    if [[ ! -f "$dockerfile" ]]; then
        log::error "Dockerfile not found: $dockerfile"
        return 1
    fi
    
    local image_name=$(basename "$(pwd)")
    log::info "Building Docker image: $image_name:$tag"
    
    local cmd="docker build -f $dockerfile -t $image_name:$tag"
    
    # Add build arguments if provided
    if [[ -n "$build_args" ]]; then
        cmd="$cmd $build_args"
    fi
    
    # Add labels for tracking
    cmd="$cmd --label build.date=$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
    cmd="$cmd --label build.version=$tag"
    cmd="$cmd --label build.user=$USER"
    
    # Execute build
    eval "$cmd $context"
}

docker::push() {
    local image="${1}"
    local registry="${2:-}"
    
    if [[ -n "$registry" ]]; then
        local full_image="$registry/$image"
        log::info "Tagging image for registry: $full_image"
        docker tag "$image" "$full_image"
        image="$full_image"
    fi
    
    log::info "Pushing image: $image"
    retry::exponential_backoff 3 5 30 docker push "$image"
}

docker::pull() {
    local image="${1}"
    log::info "Pulling image: $image"
    retry::exponential_backoff 3 5 30 docker pull "$image"
}

docker::tag() {
    local source="${1}"
    local target="${2}"
    log::info "Tagging $source as $target"
    docker tag "$source" "$target"
}

docker::image_exists() {
    local image="${1}"
    docker image inspect "$image" &>/dev/null
}

docker::prune_images() {
    log::info "Pruning unused images..."
    docker image prune -af
}

# ============================================================================
# Container Management (Netflix Container Pattern)
# ============================================================================

docker::run() {
    local image="${1}"
    local name="${2:-}"
    local options="${3:-}"
    
    local cmd="docker run"
    
    # Add name if provided
    [[ -n "$name" ]] && cmd="$cmd --name $name"
    
    # Add common options
    cmd="$cmd --rm"  # Remove container after exit
    cmd="$cmd -d"    # Run in background
    
    # Add custom options
    [[ -n "$options" ]] && cmd="$cmd $options"
    
    # Add image
    cmd="$cmd $image"
    
    log::info "Starting container: $name"
    eval "$cmd"
}

docker::stop() {
    local container="${1}"
    local timeout="${2:-10}"
    
    log::info "Stopping container: $container"
    docker stop -t "$timeout" "$container"
}

docker::restart() {
    local container="${1}"
    
    log::info "Restarting container: $container"
    docker restart "$container"
}

docker::logs() {
    local container="${1}"
    local options="${2:--f --tail 100}"
    
    docker logs $options "$container"
}

docker::exec() {
    local container="${1}"
    local command="${2}"
    local options="${3:--it}"
    
    docker exec $options "$container" $command
}

docker::container_exists() {
    local container="${1}"
    docker container inspect "$container" &>/dev/null
}

docker::container_running() {
    local container="${1}"
    [[ "$(docker container inspect -f '{{.State.Running}}' "$container" 2>/dev/null)" == "true" ]]
}

docker::list_containers() {
    local all="${1:-false}"
    
    if [[ "$all" == "true" ]]; then
        docker ps -a
    else
        docker ps
    fi
}

# ============================================================================
# Docker Compose (Microservices Orchestration Pattern)
# ============================================================================

docker::compose_up() {
    local file="${1:-docker-compose.yml}"
    local options="${2:-}"
    
    if [[ ! -f "$file" ]]; then
        log::error "Docker Compose file not found: $file"
        return 1
    fi
    
    log::info "Starting services with docker-compose..."
    docker compose -f "$file" up -d $options
}

docker::compose_down() {
    local file="${1:-docker-compose.yml}"
    local options="${2:-}"
    
    log::info "Stopping services with docker-compose..."
    docker compose -f "$file" down $options
}

docker::compose_restart() {
    local file="${1:-docker-compose.yml}"
    local service="${2:-}"
    
    if [[ -n "$service" ]]; then
        log::info "Restarting service: $service"
        docker compose -f "$file" restart "$service"
    else
        log::info "Restarting all services..."
        docker compose -f "$file" restart
    fi
}

docker::compose_logs() {
    local file="${1:-docker-compose.yml}"
    local service="${2:-}"
    local options="${3:--f --tail 100}"
    
    if [[ -n "$service" ]]; then
        docker compose -f "$file" logs $options "$service"
    else
        docker compose -f "$file" logs $options
    fi
}

docker::compose_ps() {
    local file="${1:-docker-compose.yml}"
    docker compose -f "$file" ps
}

docker::compose_build() {
    local file="${1:-docker-compose.yml}"
    local service="${2:-}"
    local options="${3:-}"
    
    if [[ -n "$service" ]]; then
        log::info "Building service: $service"
        docker compose -f "$file" build $options "$service"
    else
        log::info "Building all services..."
        docker compose -f "$file" build $options
    fi
}

# ============================================================================
# Volume Management (Persistent Storage Pattern)
# ============================================================================

docker::volume_create() {
    local name="${1}"
    local driver="${2:-local}"
    
    log::info "Creating volume: $name"
    docker volume create --driver "$driver" "$name"
}

docker::volume_remove() {
    local name="${1}"
    
    log::info "Removing volume: $name"
    docker volume rm "$name"
}

docker::volume_exists() {
    local name="${1}"
    docker volume inspect "$name" &>/dev/null
}

docker::volume_list() {
    docker volume ls
}

docker::volume_prune() {
    log::info "Pruning unused volumes..."
    docker volume prune -f
}

# ============================================================================
# Network Management (Service Mesh Pattern)
# ============================================================================

docker::network_create() {
    local name="${1}"
    local driver="${2:-bridge}"
    local subnet="${3:-}"
    
    log::info "Creating network: $name"
    
    local cmd="docker network create --driver $driver"
    [[ -n "$subnet" ]] && cmd="$cmd --subnet $subnet"
    cmd="$cmd $name"
    
    eval "$cmd"
}

docker::network_remove() {
    local name="${1}"
    
    log::info "Removing network: $name"
    docker network rm "$name"
}

docker::network_exists() {
    local name="${1}"
    docker network inspect "$name" &>/dev/null
}

docker::network_list() {
    docker network ls
}

docker::network_connect() {
    local network="${1}"
    local container="${2}"
    
    log::info "Connecting $container to network $network"
    docker network connect "$network" "$container"
}

docker::network_disconnect() {
    local network="${1}"
    local container="${2}"
    
    log::info "Disconnecting $container from network $network"
    docker network disconnect "$network" "$container"
}

# ============================================================================
# Registry Operations (Artifact Management Pattern)
# ============================================================================

docker::registry_login() {
    local registry="${1}"
    local username="${2:-}"
    local password="${3:-}"
    
    log::info "Logging into registry: $registry"
    
    if [[ -n "$username" ]] && [[ -n "$password" ]]; then
        echo "$password" | docker login "$registry" -u "$username" --password-stdin
    else
        docker login "$registry"
    fi
}

docker::registry_logout() {
    local registry="${1:-}"
    
    if [[ -n "$registry" ]]; then
        log::info "Logging out of registry: $registry"
        docker logout "$registry"
    else
        log::info "Logging out of all registries"
        docker logout
    fi
}

# ============================================================================
# Health Checks (Kubernetes Readiness Pattern)
# ============================================================================

docker::healthcheck() {
    local container="${1}"
    
    local health=$(docker container inspect -f '{{.State.Health.Status}}' "$container" 2>/dev/null)
    
    case "$health" in
        healthy)
            log::info "Container $container is healthy"
            return 0
            ;;
        unhealthy)
            log::error "Container $container is unhealthy"
            return 1
            ;;
        starting)
            log::warn "Container $container health check is starting"
            return 2
            ;;
        *)
            log::warn "Container $container has no health check"
            return 3
            ;;
    esac
}

docker::wait_healthy() {
    local container="${1}"
    local timeout="${2:-60}"
    
    log::info "Waiting for container $container to be healthy..."
    
    local elapsed=0
    while [[ $elapsed -lt $timeout ]]; do
        if docker::healthcheck "$container"; then
            log::info "Container is healthy!"
            return 0
        fi
        sleep 2
        ((elapsed += 2))
    done
    
    log::error "Container did not become healthy within ${timeout}s"
    return 1
}

# ============================================================================
# Cleanup Operations (Resource Management Pattern)
# ============================================================================

docker::cleanup() {
    log::info "Cleaning up Docker resources..."
    
    # Stop all running containers
    local containers=$(docker ps -q)
    if [[ -n "$containers" ]]; then
        log::info "Stopping running containers..."
        docker stop $containers
    fi
    
    # Remove stopped containers
    docker container prune -f
    
    # Remove unused images
    docker image prune -af
    
    # Remove unused volumes
    docker volume prune -f
    
    # Remove unused networks
    docker network prune -f
    
    # System prune for everything else
    docker system prune -af --volumes
    
    log::info "Docker cleanup complete"
}

docker::stats() {
    local format="${1:-table}"
    
    case "$format" in
        json)
            docker stats --no-stream --format "json"
            ;;
        simple)
            docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}"
            ;;
        *)
            docker stats --no-stream
            ;;
    esac
}

# ============================================================================
# Security Scanning (DevSecOps Pattern)
# ============================================================================

docker::security_scan() {
    local image="${1:-}"
    
    # Try different scanners in order of preference
    if command -v trivy &>/dev/null; then
        log::info "Scanning with Trivy..."
        trivy image "$image"
    elif command -v snyk &>/dev/null; then
        log::info "Scanning with Snyk..."
        snyk container test "$image"
    elif command -v grype &>/dev/null; then
        log::info "Scanning with Grype..."
        grype "$image"
    else
        log::warn "No security scanner found. Install trivy, snyk, or grype."
        return 1
    fi
}