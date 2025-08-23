import SwiftUI

@main
struct AiCanAutomationApp: App {
    @StateObject private var configManager = ConfigurationManager()
    @StateObject private var automationService = AutomationService()
    
    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(configManager)
                .environmentObject(automationService)
                .preferredColorScheme(.light)
        }
    }
}

struct ContentView: View {
    @EnvironmentObject var configManager: ConfigurationManager
    @EnvironmentObject var automationService: AutomationService
    @State private var selectedTab = 0
    
    var body: some View {
        TabView(selection: $selectedTab) {
            DashboardView()
                .tabItem {
                    Image(systemName: "command")
                    Text("Dashboard")
                }
                .tag(0)
            
            NotionView()
                .tabItem {
                    Image(systemName: "doc.text")
                    Text("Notion")
                }
                .tag(1)
            
            LogsView()
                .tabItem {
                    Image(systemName: "text.alignleft")
                    Text("Logs")
                }
                .tag(2)
            
            SettingsView()
                .tabItem {
                    Image(systemName: "gear")
                    Text("Settings")
                }
                .tag(3)
        }
        .onAppear {
            configManager.loadConfiguration()
        }
    }
}

#Preview {
    ContentView()
        .environmentObject(ConfigurationManager())
        .environmentObject(AutomationService())
}