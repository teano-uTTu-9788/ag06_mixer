import SwiftUI

struct SongLibraryView: View {
    @EnvironmentObject var songManager: SongManager
    @EnvironmentObject var audioEngine: AiOkeAudioEngine
    
    @State private var showingGenreFilter = false
    
    var body: some View {
        NavigationView {
            VStack(spacing: 0) {
                // Search and filter bar
                searchAndFilterBar
                
                // Songs list
                songsListView
            }
            .navigationTitle("Song Library")
            .navigationBarTitleDisplayMode(.large)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Filter") {
                        showingGenreFilter = true
                    }
                }
            }
            .actionSheet(isPresented: $showingGenreFilter) {
                ActionSheet(
                    title: Text("Filter by Genre"),
                    buttons: genreFilterButtons
                )
            }
            .onChange(of: songManager.searchText) { _ in
                songManager.updateFilteredSongs()
            }
            .onChange(of: songManager.selectedGenre) { _ in
                songManager.updateFilteredSongs()
            }
        }
    }
    
    // MARK: - Search and Filter Bar
    private var searchAndFilterBar: some View {
        VStack(spacing: 12) {
            // Search bar
            HStack {
                Image(systemName: "magnifyingglass")
                    .foregroundColor(.gray)
                
                TextField("Search songs or artists...", text: $songManager.searchText)
                    .textFieldStyle(PlainTextFieldStyle())
                
                if !songManager.searchText.isEmpty {
                    Button("Clear") {
                        songManager.searchText = ""
                    }
                    .font(.caption)
                    .foregroundColor(.purple)
                }
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 12)
            .background(Color(UIColor.systemGray6))
            .cornerRadius(10)
            .padding(.horizontal)
            
            // Active filter indicator
            if songManager.selectedGenre != "All" {
                HStack {
                    Text("Filtered by: \(songManager.selectedGenre)")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    
                    Spacer()
                    
                    Button("Clear") {
                        songManager.selectedGenre = "All"
                    }
                    .font(.caption)
                    .foregroundColor(.purple)
                }
                .padding(.horizontal)
            }
        }
        .padding(.top, 8)
    }
    
    // MARK: - Songs List
    private var songsListView: some View {
        List {
            ForEach(songManager.filteredSongs) { song in
                SongRowView(song: song) {
                    loadAndPlaySong(song)
                }
            }
        }
        .listStyle(PlainListStyle())
    }
    
    // MARK: - Genre Filter Buttons
    private var genreFilterButtons: [ActionSheet.Button] {
        var buttons: [ActionSheet.Button] = []
        
        for genre in songManager.availableGenres {
            buttons.append(.default(Text(genre)) {
                songManager.selectedGenre = genre
            })
        }
        
        buttons.append(.cancel())
        return buttons
    }
    
    // MARK: - Actions
    private func loadAndPlaySong(_ song: Song) {
        audioEngine.stop() // Stop current playback
        audioEngine.loadSong(song)
        
        // Switch to main karaoke view (this would need navigation coordination)
        // For now, just load the song
    }
}

// MARK: - Song Row View
struct SongRowView: View {
    let song: Song
    let onTap: () -> Void
    
    @EnvironmentObject var songManager: SongManager
    
    var body: some View {
        Button(action: onTap) {
            HStack(spacing: 16) {
                // Song artwork placeholder
                RoundedRectangle(cornerRadius: 8)
                    .fill(LinearGradient(
                        colors: [.purple.opacity(0.3), .blue.opacity(0.3)],
                        startPoint: .topLeading,
                        endPoint: .bottomTrailing
                    ))
                    .frame(width: 60, height: 60)
                    .overlay {
                        Image(systemName: "music.note")
                            .foregroundColor(.purple)
                            .font(.title3)
                    }
                
                // Song info
                VStack(alignment: .leading, spacing: 4) {
                    Text(song.title)
                        .font(.headline)
                        .foregroundColor(.primary)
                        .lineLimit(1)
                    
                    Text(song.artist)
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                        .lineLimit(1)
                    
                    HStack(spacing: 12) {
                        Text(song.genre)
                            .font(.caption)
                            .padding(.horizontal, 8)
                            .padding(.vertical, 2)
                            .background(Color.purple.opacity(0.1))
                            .foregroundColor(.purple)
                            .cornerRadius(4)
                        
                        Text(song.durationFormatted)
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                }
                
                Spacer()
                
                // Favorite button
                Button(action: {
                    songManager.toggleFavorite(for: song)
                }) {
                    Image(systemName: songManager.isFavorite(song) ? "heart.fill" : "heart")
                        .foregroundColor(songManager.isFavorite(song) ? .red : .gray)
                        .font(.title3)
                }
                .buttonStyle(PlainButtonStyle())
            }
            .padding(.vertical, 8)
        }
        .buttonStyle(PlainButtonStyle())
    }
}

// MARK: - Song Picker View (for sheet presentation)
struct SongPickerView: View {
    @Binding var selectedSong: Song?
    @Environment(\.dismiss) private var dismiss
    
    var body: some View {
        NavigationView {
            SongLibraryView()
                .navigationTitle("Select Song")
                .navigationBarTitleDisplayMode(.inline)
                .toolbar {
                    ToolbarItem(placement: .navigationBarLeading) {
                        Button("Cancel") {
                            dismiss()
                        }
                    }
                }
        }
    }
}

#Preview {
    SongLibraryView()
        .environmentObject(SongManager())
        .environmentObject(AiOkeAudioEngine())
}