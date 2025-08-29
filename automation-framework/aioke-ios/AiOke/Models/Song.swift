import Foundation
import AVFoundation

// MARK: - Song Model
struct Song: Identifiable, Codable {
    let id = UUID()
    let title: String
    let artist: String
    let duration: TimeInterval
    let fileName: String
    let genre: String
    let releaseYear: Int?
    var isFavorite: Bool = false
    
    // Computed properties
    var durationFormatted: String {
        let minutes = Int(duration) / 60
        let seconds = Int(duration) % 60
        return String(format: "%d:%02d", minutes, seconds)
    }
    
    var audioURL: URL? {
        guard let url = Bundle.main.url(forResource: fileName, withExtension: "mp3") else {
            return nil
        }
        return url
    }
}

// MARK: - Song Manager
class SongManager: ObservableObject {
    @Published var songs: [Song] = []
    @Published var filteredSongs: [Song] = []
    @Published var searchText: String = ""
    @Published var selectedGenre: String = "All"
    @Published var favorites: Set<UUID> = []
    
    private let favoritesKey = "AiOkeFavorites"
    
    init() {
        loadDemoSongs()
        loadFavorites()
        updateFilteredSongs()
    }
    
    // MARK: - Demo Songs (Royalty-free for App Store)
    private func loadDemoSongs() {
        songs = [
            Song(title: "Electric Dreams", 
                 artist: "Demo Artist", 
                 duration: 180, 
                 fileName: "electric_dreams", 
                 genre: "Electronic", 
                 releaseYear: 2024),
            
            Song(title: "Sunset Boulevard", 
                 artist: "Demo Artist", 
                 duration: 210, 
                 fileName: "sunset_boulevard", 
                 genre: "Pop", 
                 releaseYear: 2024),
            
            Song(title: "Mountain Echo", 
                 artist: "Demo Artist", 
                 duration: 195, 
                 fileName: "mountain_echo", 
                 genre: "Folk", 
                 releaseYear: 2024),
            
            Song(title: "City Lights", 
                 artist: "Demo Artist", 
                 duration: 165, 
                 fileName: "city_lights", 
                 genre: "Jazz", 
                 releaseYear: 2024),
            
            Song(title: "Ocean Waves", 
                 artist: "Demo Artist", 
                 duration: 220, 
                 fileName: "ocean_waves", 
                 genre: "Ambient", 
                 releaseYear: 2024),
            
            Song(title: "Rock Anthem", 
                 artist: "Demo Artist", 
                 duration: 190, 
                 fileName: "rock_anthem", 
                 genre: "Rock", 
                 releaseYear: 2024),
            
            Song(title: "Country Road", 
                 artist: "Demo Artist", 
                 duration: 175, 
                 fileName: "country_road", 
                 genre: "Country", 
                 releaseYear: 2024),
            
            Song(title: "Hip Hop Flow", 
                 artist: "Demo Artist", 
                 duration: 155, 
                 fileName: "hip_hop_flow", 
                 genre: "Hip Hop", 
                 releaseYear: 2024)
        ]
    }
    
    // MARK: - Filtering and Search
    func updateFilteredSongs() {
        filteredSongs = songs.filter { song in
            let matchesSearch = searchText.isEmpty || 
                               song.title.localizedCaseInsensitiveContains(searchText) ||
                               song.artist.localizedCaseInsensitiveContains(searchText)
            
            let matchesGenre = selectedGenre == "All" || song.genre == selectedGenre
            
            return matchesSearch && matchesGenre
        }
    }
    
    var availableGenres: [String] {
        ["All"] + Array(Set(songs.map { $0.genre })).sorted()
    }
    
    // MARK: - Favorites Management
    func toggleFavorite(for song: Song) {
        if favorites.contains(song.id) {
            favorites.remove(song.id)
        } else {
            favorites.insert(song.id)
        }
        saveFavorites()
    }
    
    func isFavorite(_ song: Song) -> Bool {
        favorites.contains(song.id)
    }
    
    private func saveFavorites() {
        if let data = try? JSONEncoder().encode(Array(favorites)) {
            UserDefaults.standard.set(data, forKey: favoritesKey)
        }
    }
    
    private func loadFavorites() {
        guard let data = UserDefaults.standard.data(forKey: favoritesKey),
              let favoriteIds = try? JSONDecoder().decode([UUID].self, from: data) else {
            return
        }
        favorites = Set(favoriteIds)
    }
}

// MARK: - Song Extensions
extension Song {
    static let demo = Song(
        title: "Demo Song",
        artist: "Demo Artist",
        duration: 180,
        fileName: "demo_song",
        genre: "Pop",
        releaseYear: 2024
    )
}