#!/usr/bin/env bash
# CI/CD Library - Continuous Integration Pattern
# Provides CI/CD automation utilities

# Require core library
deps::require "core"

# ============================================================================
# CI Environment Detection (GitHub Actions Pattern)
# ============================================================================

ci::is_ci() {
    [[ "${CI:-false}" == "true" ]] || \
    [[ -n "${GITHUB_ACTIONS:-}" ]] || \
    [[ -n "${JENKINS_HOME:-}" ]] || \
    [[ -n "${GITLAB_CI:-}" ]] || \
    [[ -n "${CIRCLECI:-}" ]] || \
    [[ -n "${TRAVIS:-}" ]]
}

ci::detect_provider() {
    if [[ -n "${GITHUB_ACTIONS:-}" ]]; then
        echo "github"
    elif [[ -n "${GITLAB_CI:-}" ]]; then
        echo "gitlab"
    elif [[ -n "${JENKINS_HOME:-}" ]]; then
        echo "jenkins"
    elif [[ -n "${CIRCLECI:-}" ]]; then
        echo "circle"
    elif [[ -n "${TRAVIS:-}" ]]; then
        echo "travis"
    else
        echo "unknown"
    fi
}

ci::is_pull_request() {
    [[ "${GITHUB_EVENT_NAME:-}" == "pull_request" ]] || \
    [[ -n "${CIRCLE_PULL_REQUEST:-}" ]] || \
    [[ "${TRAVIS_PULL_REQUEST:-false}" != "false" ]] || \
    [[ -n "${CI_MERGE_REQUEST_ID:-}" ]]
}

ci::get_branch() {
    # GitHub Actions
    if [[ -n "${GITHUB_REF:-}" ]]; then
        echo "${GITHUB_REF#refs/heads/}"
    # GitLab CI
    elif [[ -n "${CI_COMMIT_BRANCH:-}" ]]; then
        echo "$CI_COMMIT_BRANCH"
    # Circle CI
    elif [[ -n "${CIRCLE_BRANCH:-}" ]]; then
        echo "$CIRCLE_BRANCH"
    # Travis CI
    elif [[ -n "${TRAVIS_BRANCH:-}" ]]; then
        echo "$TRAVIS_BRANCH"
    # Fallback to git
    else
        git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown"
    fi
}

ci::get_commit() {
    # GitHub Actions
    if [[ -n "${GITHUB_SHA:-}" ]]; then
        echo "$GITHUB_SHA"
    # GitLab CI
    elif [[ -n "${CI_COMMIT_SHA:-}" ]]; then
        echo "$CI_COMMIT_SHA"
    # Circle CI
    elif [[ -n "${CIRCLE_SHA1:-}" ]]; then
        echo "$CIRCLE_SHA1"
    # Travis CI
    elif [[ -n "${TRAVIS_COMMIT:-}" ]]; then
        echo "$TRAVIS_COMMIT"
    # Fallback to git
    else
        git rev-parse HEAD 2>/dev/null || echo "unknown"
    fi
}

# ============================================================================
# GitHub Actions Specific (GitHub Workflow Pattern)
# ============================================================================

github::set_output() {
    local name="$1"
    local value="$2"
    
    if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
        echo "${name}=${value}" >> "$GITHUB_OUTPUT"
    else
        echo "::set-output name=${name}::${value}"
    fi
}

github::set_env() {
    local name="$1"
    local value="$2"
    
    if [[ -n "${GITHUB_ENV:-}" ]]; then
        echo "${name}=${value}" >> "$GITHUB_ENV"
    else
        echo "::set-env name=${name}::${value}"
    fi
    
    # Also set for current session
    export "$name=$value"
}

github::add_path() {
    local path="$1"
    
    if [[ -n "${GITHUB_PATH:-}" ]]; then
        echo "$path" >> "$GITHUB_PATH"
    else
        echo "::add-path::${path}"
    fi
    
    # Also add for current session
    export PATH="${path}:$PATH"
}

github::group() {
    local title="$1"
    echo "::group::${title}"
}

github::endgroup() {
    echo "::endgroup::"
}

github::error() {
    local message="$1"
    local file="${2:-}"
    local line="${3:-}"
    
    local output="::error"
    [[ -n "$file" ]] && output="${output} file=${file}"
    [[ -n "$line" ]] && output="${output},line=${line}"
    echo "${output}::${message}"
}

github::warning() {
    local message="$1"
    local file="${2:-}"
    local line="${3:-}"
    
    local output="::warning"
    [[ -n "$file" ]] && output="${output} file=${file}"
    [[ -n "$line" ]] && output="${output},line=${line}"
    echo "${output}::${message}"
}

github::notice() {
    local message="$1"
    echo "::notice::${message}"
}

github::add_mask() {
    local value="$1"
    echo "::add-mask::${value}"
}

# ============================================================================
# Artifact Management (Artifact Storage Pattern)
# ============================================================================

ci::upload_artifact() {
    local name="$1"
    local path="$2"
    
    case "$(ci::detect_provider)" in
        github)
            # GitHub Actions artifact upload
            if command -v actions-upload-artifact &>/dev/null; then
                actions-upload-artifact --name "$name" --path "$path"
            else
                log::warn "GitHub Actions artifact upload not available"
            fi
            ;;
        gitlab)
            # GitLab CI artifacts are configured in .gitlab-ci.yml
            log::info "Artifacts configured in .gitlab-ci.yml"
            ;;
        *)
            # Generic artifact storage
            local artifact_dir="${CI_ARTIFACTS_DIR:-./artifacts}"
            mkdir -p "$artifact_dir"
            cp -r "$path" "$artifact_dir/$name"
            log::info "Artifact stored in $artifact_dir/$name"
            ;;
    esac
}

ci::download_artifact() {
    local name="$1"
    local path="${2:-.}"
    
    case "$(ci::detect_provider)" in
        github)
            if command -v actions-download-artifact &>/dev/null; then
                actions-download-artifact --name "$name" --path "$path"
            else
                log::warn "GitHub Actions artifact download not available"
            fi
            ;;
        *)
            local artifact_dir="${CI_ARTIFACTS_DIR:-./artifacts}"
            if [[ -d "$artifact_dir/$name" ]]; then
                cp -r "$artifact_dir/$name" "$path"
                log::info "Artifact downloaded from $artifact_dir/$name"
            else
                log::error "Artifact not found: $name"
                return 1
            fi
            ;;
    esac
}

# ============================================================================
# Cache Management (Build Cache Pattern)
# ============================================================================

ci::cache_key() {
    local prefix="$1"
    local files=("${@:2}")
    
    local hash=""
    for file in "${files[@]}"; do
        if [[ -f "$file" ]]; then
            hash="${hash}$(sha256sum "$file" | cut -d' ' -f1)"
        fi
    done
    
    if [[ -z "$hash" ]]; then
        hash=$(date +%Y%m%d)
    else
        hash=$(echo "$hash" | sha256sum | cut -d' ' -f1)
    fi
    
    echo "${prefix}-${hash}"
}

ci::restore_cache() {
    local key="$1"
    local path="$2"
    
    case "$(ci::detect_provider)" in
        github)
            github::group "Restore cache: $key"
            # GitHub Actions cache action would handle this
            log::info "Cache restore handled by GitHub Actions"
            github::endgroup
            ;;
        *)
            local cache_dir="${CI_CACHE_DIR:-${HOME}/.ci-cache}"
            local cache_path="$cache_dir/$key"
            
            if [[ -d "$cache_path" ]]; then
                log::info "Restoring cache from $cache_path"
                cp -r "$cache_path/"* "$path/" 2>/dev/null || true
                return 0
            else
                log::info "Cache not found for key: $key"
                return 1
            fi
            ;;
    esac
}

ci::save_cache() {
    local key="$1"
    local path="$2"
    
    case "$(ci::detect_provider)" in
        github)
            github::group "Save cache: $key"
            # GitHub Actions cache action would handle this
            log::info "Cache save handled by GitHub Actions"
            github::endgroup
            ;;
        *)
            local cache_dir="${CI_CACHE_DIR:-${HOME}/.ci-cache}"
            local cache_path="$cache_dir/$key"
            
            log::info "Saving cache to $cache_path"
            mkdir -p "$cache_path"
            cp -r "$path/"* "$cache_path/" 2>/dev/null || true
            ;;
    esac
}

# ============================================================================
# Deployment Functions (Deploy Pattern)
# ============================================================================

ci::deploy_staging() {
    log::info "Deploying to staging environment..."
    
    # Pre-deployment checks
    if ! ci::pre_deploy_checks "staging"; then
        log::error "Pre-deployment checks failed"
        return 1
    fi
    
    # Deploy based on project type
    if [[ -f "docker-compose.yml" ]]; then
        ci::deploy_docker "staging"
    elif [[ -f "kubernetes.yaml" ]] || [[ -d "k8s" ]]; then
        ci::deploy_kubernetes "staging"
    elif [[ -f "serverless.yml" ]]; then
        ci::deploy_serverless "staging"
    else
        ci::deploy_traditional "staging"
    fi
    
    # Post-deployment verification
    ci::post_deploy_verify "staging"
}

ci::deploy_production() {
    log::info "Deploying to production environment..."
    
    # Extra safety checks for production
    if ! ci::is_main_branch; then
        log::error "Production deployment only allowed from main branch"
        return 1
    fi
    
    # Pre-deployment checks
    if ! ci::pre_deploy_checks "production"; then
        log::error "Pre-deployment checks failed"
        return 1
    fi
    
    # Create backup before deployment
    ci::backup_current "production"
    
    # Deploy with blue-green or canary if configured
    if [[ "${DEPLOY_STRATEGY:-}" == "blue-green" ]]; then
        ci::deploy_blue_green "production"
    elif [[ "${DEPLOY_STRATEGY:-}" == "canary" ]]; then
        ci::deploy_canary "production"
    else
        ci::deploy_traditional "production"
    fi
    
    # Post-deployment verification
    if ! ci::post_deploy_verify "production"; then
        log::error "Post-deployment verification failed, rolling back..."
        ci::rollback "production"
        return 1
    fi
}

ci::is_main_branch() {
    local branch=$(ci::get_branch)
    [[ "$branch" == "main" ]] || [[ "$branch" == "master" ]] || [[ "$branch" == "production" ]]
}

ci::pre_deploy_checks() {
    local environment="$1"
    
    log::info "Running pre-deployment checks for $environment..."
    
    # Check all tests pass
    if ! cmd::test; then
        log::error "Tests failed"
        return 1
    fi
    
    # Check no security issues
    if ! cmd::security; then
        log::error "Security scan failed"
        return 1
    fi
    
    # Check deployment configuration exists
    local config_file="deploy/${environment}.env"
    if [[ ! -f "$config_file" ]]; then
        log::warn "Deployment configuration not found: $config_file"
    fi
    
    return 0
}

ci::post_deploy_verify() {
    local environment="$1"
    local health_endpoint="${2:-/health}"
    
    log::info "Verifying deployment to $environment..."
    
    # Get environment URL
    local url=$(ci::get_environment_url "$environment")
    
    if [[ -z "$url" ]]; then
        log::warn "No URL configured for environment: $environment"
        return 0
    fi
    
    # Wait for service to be healthy
    if integration::wait_for_url "${url}${health_endpoint}" 60 200; then
        log::info "Deployment verified successfully"
        return 0
    else
        log::error "Deployment verification failed"
        return 1
    fi
}

ci::get_environment_url() {
    local environment="$1"
    
    case "$environment" in
        local)
            echo "http://localhost:3000"
            ;;
        staging)
            echo "${STAGING_URL:-https://staging.example.com}"
            ;;
        production)
            echo "${PRODUCTION_URL:-https://example.com}"
            ;;
        *)
            echo ""
            ;;
    esac
}

# ============================================================================
# Deployment Strategies (Advanced Deploy Pattern)
# ============================================================================

ci::deploy_docker() {
    local environment="$1"
    
    log::info "Deploying Docker containers to $environment..."
    
    # Build and push images
    docker::build "Dockerfile" "$environment"
    
    # Deploy based on environment
    case "$environment" in
        local|staging)
            docker::compose_up "docker-compose.${environment}.yml"
            ;;
        production)
            # Use Docker Swarm or Kubernetes for production
            log::info "Production Docker deployment requires orchestration"
            ;;
    esac
}

ci::deploy_kubernetes() {
    local environment="$1"
    
    log::info "Deploying to Kubernetes cluster ($environment)..."
    
    # Apply Kubernetes manifests
    kubectl apply -f "k8s/${environment}/" --namespace="$environment"
    
    # Wait for rollout to complete
    kubectl rollout status deployment/app --namespace="$environment" --timeout=5m
}

ci::deploy_serverless() {
    local environment="$1"
    
    log::info "Deploying serverless functions to $environment..."
    
    if command -v serverless &>/dev/null; then
        serverless deploy --stage "$environment"
    else
        log::error "Serverless framework not installed"
        return 1
    fi
}

ci::deploy_traditional() {
    local environment="$1"
    
    log::info "Deploying using traditional method to $environment..."
    
    # This would typically use rsync, scp, or similar
    local deploy_script="./deploy/deploy_${environment}.sh"
    
    if [[ -x "$deploy_script" ]]; then
        "$deploy_script"
    else
        log::warn "No deployment script found: $deploy_script"
    fi
}

# ============================================================================
# Release Management (Semantic Release Pattern)
# ============================================================================

ci::create_release() {
    local version="${1:-}"
    local notes="${2:-}"
    
    # Auto-determine version if not provided
    if [[ -z "$version" ]]; then
        version=$(git::next_version)
    fi
    
    log::info "Creating release: $version"
    
    # Create git tag
    git::tag "$version" "Release $version"
    
    # Create GitHub release if available
    if command -v gh &>/dev/null && [[ "$(ci::detect_provider)" == "github" ]]; then
        if [[ -n "$notes" ]]; then
            gh release create "$version" --title "Release $version" --notes "$notes"
        else
            gh release create "$version" --title "Release $version" --generate-notes
        fi
    fi
    
    # Trigger deployment pipeline
    if ci::is_ci; then
        github::set_output "release_version" "$version"
        github::set_output "should_deploy" "true"
    fi
}

# ============================================================================
# Notification Functions (Notification Pattern)
# ============================================================================

ci::notify_slack() {
    local message="$1"
    local webhook_url="${SLACK_WEBHOOK_URL:-}"
    local channel="${2:-#deployments}"
    
    if [[ -z "$webhook_url" ]]; then
        log::debug "Slack webhook not configured"
        return 0
    fi
    
    local payload=$(cat <<EOF
{
    "channel": "$channel",
    "text": "$message",
    "username": "CI/CD Bot",
    "icon_emoji": ":rocket:"
}
EOF
)
    
    curl -X POST -H 'Content-type: application/json' \
         --data "$payload" \
         "$webhook_url" &>/dev/null
}

ci::notify_email() {
    local subject="$1"
    local body="$2"
    local recipients="${3:-${CI_EMAIL_RECIPIENTS:-}}"
    
    if [[ -z "$recipients" ]]; then
        log::debug "Email recipients not configured"
        return 0
    fi
    
    if command -v mail &>/dev/null; then
        echo "$body" | mail -s "$subject" "$recipients"
    else
        log::debug "Mail command not available"
    fi
}