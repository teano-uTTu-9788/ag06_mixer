import Foundation
import Combine

class ConfigurationManager: ObservableObject {
    @Published var configuration = Configuration()
    
    private let userDefaults = UserDefaults.standard
    private let configurationKey = "AiCanConfiguration"
    
    func loadConfiguration() {
        if let data = userDefaults.data(forKey: configurationKey),
           let decoded = try? JSONDecoder().decode(Configuration.self, from: data) {
            DispatchQueue.main.async {
                self.configuration = decoded
            }
        }
    }
    
    func saveConfiguration() {
        if let encoded = try? JSONEncoder().encode(configuration) {
            userDefaults.set(encoded, forKey: configurationKey)
        }
    }
    
    func updateGitHubSettings(token: String, repository: String) {
        configuration.githubToken = token
        configuration.githubRepository = repository
        saveConfiguration()
    }
    
    func updateNotionSettings(token: String, defaultPageId: String = "") {
        configuration.notionToken = token
        configuration.defaultNotionPageId = defaultPageId
        saveConfiguration()
    }
    
    func validateConfiguration() -> [String] {
        var errors: [String] = []
        
        if configuration.githubToken.isEmpty {
            errors.append("GitHub token is required")
        } else if !configuration.githubToken.hasPrefix("ghp_") && 
                  !configuration.githubToken.hasPrefix("github_pat_") {
            errors.append("GitHub token format appears invalid")
        }
        
        if configuration.githubRepository.isEmpty {
            errors.append("GitHub repository is required")
        } else if !configuration.githubRepository.contains("/") {
            errors.append("Repository should be in format 'owner/repo'")
        }
        
        if !configuration.notionToken.isEmpty &&
           !configuration.notionToken.hasPrefix("secret_") {
            errors.append("Notion token format appears invalid (should start with 'secret_')")
        }
        
        return errors
    }
    
    func clearConfiguration() {
        configuration = Configuration()
        userDefaults.removeObject(forKey: configurationKey)
    }
}