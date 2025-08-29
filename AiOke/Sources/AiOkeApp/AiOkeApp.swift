import SwiftUI

/// Main AiOke app entry point
@main
struct AiOkeApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
                .preferredColorScheme(.light) // Consistent appearance for MVP
        }
    }
}