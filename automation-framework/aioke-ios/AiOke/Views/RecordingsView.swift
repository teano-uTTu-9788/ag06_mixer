import SwiftUI

struct RecordingsView: View {
    @EnvironmentObject var recordingManager: RecordingManager
    @State private var showingDeleteAlert = false
    @State private var recordingToDelete: Recording?
    @State private var showingRenameSheet = false
    @State private var recordingToRename: Recording?
    @State private var newRecordingName = ""
    
    var body: some View {
        NavigationView {
            VStack {
                if recordingManager.recordings.isEmpty {
                    emptyStateView
                } else {
                    recordingsListView
                }
            }
            .navigationTitle("Recordings")
            .navigationBarTitleDisplayMode(.large)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    if !recordingManager.recordings.isEmpty {
                        EditButton()
                    }
                }
            }
            .alert("Delete Recording", isPresented: $showingDeleteAlert) {
                Button("Delete", role: .destructive) {
                    if let recording = recordingToDelete {
                        recordingManager.deleteRecording(recording)
                    }
                }
                Button("Cancel", role: .cancel) { }
            } message: {
                Text("This recording will be permanently deleted.")
            }
            .sheet(isPresented: $showingRenameSheet) {
                renameRecordingSheet
            }
        }
    }
    
    // MARK: - Empty State
    private var emptyStateView: some View {
        VStack(spacing: 24) {
            Spacer()
            
            Image(systemName: "waveform.circle")
                .font(.system(size: 80))
                .foregroundColor(.purple.opacity(0.3))
            
            VStack(spacing: 8) {
                Text("No Recordings Yet")
                    .font(.title2)
                    .fontWeight(.medium)
                
                Text("Start karaoke and tap record to save your performances")
                    .font(.subheadline)
                    .foregroundColor(.secondary)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal)
            }
            
            Spacer()
            
            // Quick tips
            VStack(alignment: .leading, spacing: 12) {
                tipRow(icon: "mic.circle", text: "Tap the record button while singing")
                tipRow(icon: "slider.horizontal.3", text: "Adjust effects for better sound")
                tipRow(icon: "square.and.arrow.up", text: "Share your recordings with friends")
            }
            .padding()
            .background(Color(UIColor.systemGray6))
            .cornerRadius(12)
            .padding(.horizontal)
            
            Spacer()
        }
    }
    
    private func tipRow(icon: String, text: String) -> some View {
        HStack(spacing: 12) {
            Image(systemName: icon)
                .foregroundColor(.purple)
                .frame(width: 20)
            
            Text(text)
                .font(.subheadline)
                .foregroundColor(.secondary)
        }
    }
    
    // MARK: - Recordings List
    private var recordingsListView: some View {
        VStack(spacing: 0) {
            // Statistics header
            statisticsHeader
            
            // Recordings list
            List {
                ForEach(recordingManager.recordings) { recording in
                    RecordingRowView(recording: recording) { action in
                        handleRecordingAction(recording, action: action)
                    }
                }
                .onDelete(perform: deleteRecordings)
            }
            .listStyle(PlainListStyle())
        }
    }
    
    // MARK: - Statistics Header
    private var statisticsHeader: some View {
        HStack {
            VStack(alignment: .leading, spacing: 4) {
                Text("\(recordingManager.totalRecordings) Recordings")
                    .font(.headline)
                
                Text("Total: \(recordingManager.formattedTotalDuration)")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            
            Spacer()
            
            if recordingManager.isRecording {
                HStack(spacing: 8) {
                    Circle()
                        .fill(.red)
                        .frame(width: 8, height: 8)
                        .opacity(0.8)
                        .scaleEffect(1.2)
                        .animation(.easeInOut(duration: 1).repeatForever(), value: recordingManager.isRecording)
                    
                    Text("Recording...")
                        .font(.caption)
                        .foregroundColor(.red)
                        .fontWeight(.medium)
                }
            }
        }
        .padding(.horizontal)
        .padding(.vertical, 12)
        .background(Color(UIColor.systemBackground))
    }
    
    // MARK: - Actions
    private func handleRecordingAction(_ recording: Recording, action: RecordingAction) {
        switch action {
        case .play:
            recordingManager.playRecording(recording)
        case .rename:
            recordingToRename = recording
            newRecordingName = recording.title
            showingRenameSheet = true
        case .delete:
            recordingToDelete = recording
            showingDeleteAlert = true
        case .share:
            shareRecording(recording)
        }
    }
    
    private func deleteRecordings(offsets: IndexSet) {
        for index in offsets {
            recordingManager.deleteRecording(recordingManager.recordings[index])
        }
    }
    
    private func shareRecording(_ recording: Recording) {
        // Share functionality would go here
        // For now, this is a placeholder
        print("Sharing recording: \(recording.title)")
    }
    
    // MARK: - Rename Sheet
    private var renameRecordingSheet: some View {
        NavigationView {
            VStack(spacing: 20) {
                TextField("Recording Name", text: $newRecordingName)
                    .textFieldStyle(RoundedBorderTextFieldStyle())
                    .padding(.horizontal)
                
                Spacer()
            }
            .padding(.top)
            .navigationTitle("Rename Recording")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button("Cancel") {
                        showingRenameSheet = false
                    }
                }
                
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Save") {
                        if let recording = recordingToRename, !newRecordingName.isEmpty {
                            recordingManager.renameRecording(recording, to: newRecordingName)
                        }
                        showingRenameSheet = false
                    }
                    .disabled(newRecordingName.isEmpty)
                }
            }
        }
    }
}

// MARK: - Recording Action Enum
enum RecordingAction {
    case play, rename, delete, share
}

// MARK: - Recording Row View
struct RecordingRowView: View {
    let recording: Recording
    let onAction: (RecordingAction) -> Void
    
    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: 16) {
                // Waveform icon
                RoundedRectangle(cornerRadius: 8)
                    .fill(LinearGradient(
                        colors: [.purple.opacity(0.3), .blue.opacity(0.3)],
                        startPoint: .topLeading,
                        endPoint: .bottomTrailing
                    ))
                    .frame(width: 50, height: 50)
                    .overlay {
                        Image(systemName: "waveform")
                            .foregroundColor(.purple)
                            .font(.title3)
                    }
                
                // Recording info
                VStack(alignment: .leading, spacing: 4) {
                    Text(recording.title)
                        .font(.headline)
                        .lineLimit(2)
                    
                    if let songTitle = recording.songTitle {
                        Text("Karaoke: \(songTitle)")
                            .font(.caption)
                            .foregroundColor(.purple)
                            .padding(.horizontal, 6)
                            .padding(.vertical, 2)
                            .background(Color.purple.opacity(0.1))
                            .cornerRadius(4)
                    }
                    
                    HStack(spacing: 12) {
                        Text(recording.formattedDuration)
                            .font(.caption)
                            .foregroundColor(.secondary)
                        
                        Text(recording.formattedDate)
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                }
                
                Spacer()
                
                // Action buttons
                VStack(spacing: 8) {
                    Button(action: { onAction(.play) }) {
                        Image(systemName: "play.circle")
                            .font(.title2)
                            .foregroundColor(.purple)
                    }
                    
                    Menu {
                        Button("Rename", action: { onAction(.rename) })
                        Button("Share", action: { onAction(.share) })
                        Divider()
                        Button("Delete", role: .destructive, action: { onAction(.delete) })
                    } label: {
                        Image(systemName: "ellipsis.circle")
                            .font(.title3)
                            .foregroundColor(.gray)
                    }
                }
            }
            .padding(.vertical, 12)
            .padding(.horizontal, 16)
            .contentShape(Rectangle())
        }
    }
}

#Preview {
    RecordingsView()
        .environmentObject(RecordingManager())
}