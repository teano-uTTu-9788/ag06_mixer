#!/usr/bin/env bash
# shellcheck shell=bash
# Notion API utilities for updating page status

# Source dependencies
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./log.sh
source "${SCRIPT_DIR}/log.sh"
# shellcheck source=./utils.sh
source "${SCRIPT_DIR}/utils.sh"

# Notion API version
readonly NOTION_API_VERSION="2022-06-28"

notion_update_status() {
  required curl
  required jq
  
  local page_id="${1:-$NOTION_PAGE_ID}"
  local status="${2:-$NOTION_STATUS_NAME}"
  
  # Validate environment
  [[ -n "$NOTION_TOKEN" ]] || die "NOTION_TOKEN not set"
  [[ -n "$page_id" ]] || die "Page ID required"
  [[ -n "$status" ]] || die "Status required"
  
  log_info "Updating Notion page status: $status"
  
  # Build request payload
  local payload
  payload=$(jq -n --arg s "$status" '{
    properties: {
      Status: {
        status: {
          name: $s
        }
      }
    }
  }')
  
  # Make API request
  local response
  response=$(curl -sS -X PATCH "https://api.notion.com/v1/pages/${page_id}" \
    -H "Authorization: Bearer ${NOTION_TOKEN}" \
    -H "Notion-Version: ${NOTION_API_VERSION}" \
    -H "Content-Type: application/json" \
    --data "$payload")
  
  # Check response
  if echo "$response" | jq -e '.object=="page"' >/dev/null; then
    log_ok "Notion status updated to: $status"
  else
    log_error "Notion update failed"
    echo "$response" | jq -r '.message // .error // .' >&2
    return 1
  fi
}

notion_update_property() {
  required curl
  required jq
  
  local page_id="${1:-$NOTION_PAGE_ID}"
  local property_name="$2"
  local property_value="$3"
  local property_type="${4:-rich_text}"  # rich_text, number, checkbox, select, etc.
  
  # Validate environment
  [[ -n "$NOTION_TOKEN" ]] || die "NOTION_TOKEN not set"
  [[ -n "$page_id" ]] || die "Page ID required"
  [[ -n "$property_name" ]] || die "Property name required"
  
  log_info "Updating Notion property: $property_name = $property_value"
  
  # Build property payload based on type
  local property_payload
  case "$property_type" in
    rich_text)
      property_payload=$(jq -n --arg v "$property_value" '{
        rich_text: [{
          text: {
            content: $v
          }
        }]
      }')
      ;;
    number)
      property_payload=$(jq -n --arg v "$property_value" '{
        number: ($v | tonumber)
      }')
      ;;
    checkbox)
      property_payload=$(jq -n --arg v "$property_value" '{
        checkbox: ($v == "true")
      }')
      ;;
    select)
      property_payload=$(jq -n --arg v "$property_value" '{
        select: {
          name: $v
        }
      }')
      ;;
    *)
      die "Unsupported property type: $property_type"
      ;;
  esac
  
  # Build full payload
  local payload
  payload=$(jq -n --arg name "$property_name" --argjson prop "$property_payload" '{
    properties: {
      ($name): $prop
    }
  }')
  
  # Make API request
  local response
  response=$(curl -sS -X PATCH "https://api.notion.com/v1/pages/${page_id}" \
    -H "Authorization: Bearer ${NOTION_TOKEN}" \
    -H "Notion-Version: ${NOTION_API_VERSION}" \
    -H "Content-Type: application/json" \
    --data "$payload")
  
  # Check response
  if echo "$response" | jq -e '.object=="page"' >/dev/null; then
    log_ok "Notion property updated: $property_name"
  else
    log_error "Notion update failed"
    echo "$response" | jq -r '.message // .error // .' >&2
    return 1
  fi
}

notion_create_page() {
  required curl
  required jq
  
  local parent_id="${1:-$NOTION_DATABASE_ID}"
  local title="$2"
  local content="${3:-}"
  
  # Validate environment
  [[ -n "$NOTION_TOKEN" ]] || die "NOTION_TOKEN not set"
  [[ -n "$parent_id" ]] || die "Parent database ID required"
  [[ -n "$title" ]] || die "Title required"
  
  log_info "Creating Notion page: $title"
  
  # Build payload
  local payload
  payload=$(jq -n \
    --arg parent "$parent_id" \
    --arg title "$title" \
    --arg content "$content" '{
    parent: {
      database_id: $parent
    },
    properties: {
      Name: {
        title: [{
          text: {
            content: $title
          }
        }]
      }
    },
    children: (if $content != "" then [{
      object: "block",
      type: "paragraph",
      paragraph: {
        rich_text: [{
          type: "text",
          text: {
            content: $content
          }
        }]
      }
    }] else [] end)
  }')
  
  # Make API request
  local response
  response=$(curl -sS -X POST "https://api.notion.com/v1/pages" \
    -H "Authorization: Bearer ${NOTION_TOKEN}" \
    -H "Notion-Version: ${NOTION_API_VERSION}" \
    -H "Content-Type: application/json" \
    --data "$payload")
  
  # Check response
  if echo "$response" | jq -e '.object=="page"' >/dev/null; then
    local page_url
    page_url=$(echo "$response" | jq -r '.url')
    log_ok "Notion page created: $page_url"
    echo "$response" | jq -r '.id'
  else
    log_error "Notion page creation failed"
    echo "$response" | jq -r '.message // .error // .' >&2
    return 1
  fi
}