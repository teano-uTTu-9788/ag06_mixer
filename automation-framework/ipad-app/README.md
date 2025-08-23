# AiCan Automation iPad App

A SwiftUI-based iPad application for remotely controlling macOS terminal automation workflows through GitHub Actions and Notion integration.

## Features

### 📱 **Dashboard**
- Quick action buttons for common automation tasks (doctor, CI, test, lint, format, bootstrap)
- Real-time connection status for GitHub and Notion
- Workflow run history with status tracking
- Optional Notion page ID input for task-specific operations

### 📄 **Notion Integration**  
- Browse and manage Notion pages
- Update page statuses with quick actions
- Create new pages with templates
- Connection status monitoring

### 📋 **Logs**
- Real-time automation logs with level filtering (Info, Success, Warning, Error)
- Search functionality across all log entries
- Export logs in multiple formats (TXT, JSON, CSV)
- Auto-scroll option for live monitoring

### ⚙️ **Settings**
- GitHub configuration (Personal Access Token, Repository)
- Notion integration setup (API Token, Default Page)
- Connection testing and validation
- Advanced preferences and debug options

## Setup Instructions

### Prerequisites
- Xcode 15.0 or later
- iOS 17.0+ / iPadOS 17.0+
- Active GitHub repository with Actions enabled
- Notion workspace (optional)

### 1. GitHub Configuration
1. Create a Personal Access Token:
   - Go to GitHub Settings → Developer settings → Personal access tokens
   - Generate new token with `repo` and `workflow` scopes
   - Copy the token (starts with `ghp_` or `github_pat_`)

2. Repository setup:
   - Ensure your repository has the automation framework installed
   - Verify GitHub Actions are enabled
   - Format: `owner/repository-name`

### 2. Notion Setup (Optional)
1. Create a Notion integration:
   - Visit [notion.so/my-integrations](https://notion.so/my-integrations)
   - Create a new integration
   - Copy the Internal Integration Token (starts with `secret_`)

2. Share pages with integration:
   - Open your Notion database/page
   - Click Share → Add people → Select your integration

### 3. App Configuration
1. Launch the app on your iPad
2. Navigate to Settings tab
3. Configure GitHub integration:
   - Enter your Personal Access Token
   - Enter repository in `owner/repo` format
4. Configure Notion (optional):
   - Enter your Integration Token
   - Set default page ID for quick updates
5. Test connections to verify setup

## Usage

### Running Automation Tasks
1. Open Dashboard tab
2. Ensure GitHub shows "Connected" status
3. Optionally enter a Notion Page ID
4. Tap any automation task button:
   - **System Health Check**: Runs comprehensive system diagnostics
   - **Run CI Suite**: Executes full continuous integration pipeline
   - **Run Tests**: Runs project test suite
   - **Lint Code**: Performs code linting and style checks
   - **Format Code**: Auto-formats code according to style guide
   - **Bootstrap Environment**: Sets up development environment

### Managing Notion Pages
1. Open Notion tab
2. Update page status:
   - Enter page ID
   - Select status (Not started, In progress, Completed, Blocked, Cancelled)
   - Tap "Update Status"
3. Create new pages using quick action templates
4. Use default page for rapid status updates

### Monitoring Logs
1. Open Logs tab
2. Filter by log level using chips
3. Search logs using the search bar
4. Export logs for analysis or sharing
5. Enable auto-scroll for live monitoring

## Architecture

### App Structure
```
AiCanAutomation/
├── AiCanAutomationApp.swift          # App entry point with tab navigation
├── Models/
│   └── Configuration.swift           # Data models and enums
├── Services/
│   ├── AutomationService.swift       # GitHub Actions and Notion API client
│   └── ConfigurationManager.swift    # Settings persistence
└── Views/
    ├── DashboardView.swift           # Main automation interface
    ├── NotionView.swift              # Notion page management
    ├── LogsView.swift                # Log viewing and export
    └── SettingsView.swift            # Configuration management
```

### Key Components

#### AutomationService
- GitHub API integration for workflow dispatch
- Notion API client for page updates
- Connection testing and validation
- Log management and history

#### ConfigurationManager
- UserDefaults-based settings persistence
- Configuration validation
- Secure token storage

#### Models
- `AutomationTask`: Available automation commands
- `NotionStatus`: Page status options
- `WorkflowRun`: Execution tracking
- `LogEntry`: Structured logging

## Development

### Building
1. Open project in Xcode
2. Select iPad target
3. Build and run (⌘R)

### Testing
- Unit tests included for core functionality
- Use iOS Simulator for development
- Test on actual iPad for full functionality

### Customization
- Add new automation tasks in `Configuration.swift`
- Extend Notion integration in `NotionView.swift`
- Customize UI themes and colors
- Add new export formats in `LogsView.swift`

## Security

- API tokens stored securely using UserDefaults
- HTTPS-only communication with APIs
- Input validation for all user data
- No sensitive information logged

## Troubleshooting

### Connection Issues
1. Verify internet connectivity
2. Check API token validity
3. Ensure repository has Actions enabled
4. Verify Notion integration permissions

### Common Errors
- **401 Unauthorized**: Invalid or expired tokens
- **404 Not Found**: Repository doesn't exist or no access
- **403 Forbidden**: Insufficient permissions

### Debug Mode
Enable debug mode in Advanced Settings for detailed logging and diagnostics.

## Version History

### v1.0.0
- Initial release
- Full GitHub Actions integration
- Notion API support
- Comprehensive logging system
- Multi-format log export
- iPad-optimized interface

## Support

For issues and feature requests, please check the main automation framework documentation or create an issue in the GitHub repository.