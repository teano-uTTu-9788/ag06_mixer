import SwiftUI

struct SubscriptionView: View {
    @EnvironmentObject var configManager: ConfigurationManager
    @Environment(\.dismiss) private var dismiss
    
    @State private var selectedTier: SubscriptionTier = .pro
    @State private var showingPurchaseFlow = false
    @State private var isProcessingPurchase = false
    
    private let currentTier: SubscriptionTier = .free // In real app, get from subscription manager
    
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 24) {
                    // Header
                    headerSection
                    
                    // Current subscription status
                    currentStatusSection
                    
                    // Subscription tiers
                    subscriptionTiersSection
                    
                    // Feature comparison
                    featureComparisonSection
                    
                    // Upgrade button
                    if currentTier != .studio {
                        upgradeSection
                    }
                    
                    // Fine print
                    legalSection
                }
                .padding()
            }
            .navigationTitle("Subscription")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Done") {
                        dismiss()
                    }
                }
            }
        }
        .sheet(isPresented: $showingPurchaseFlow) {
            PurchaseFlowView(selectedTier: selectedTier)
        }
    }
    
    // MARK: - Header Section
    private var headerSection: some View {
        VStack(spacing: 16) {
            Image(systemName: "waveform.path.ecg.rectangle")
                .font(.system(size: 60))
                .foregroundColor(.accentColor)
            
            VStack(spacing: 8) {
                Text("Unlock Pro Features")
                    .font(.title2)
                    .fontWeight(.bold)
                    .multilineTextAlignment(.center)
                
                Text("Get advanced audio processing, AI-powered mixing, and professional-grade tools")
                    .font(.subheadline)
                    .foregroundColor(.secondary)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal)
            }
        }
    }
    
    // MARK: - Current Status
    private var currentStatusSection: some View {
        HStack {
            VStack(alignment: .leading) {
                Text("Current Plan")
                    .font(.subheadline)
                    .fontWeight(.medium)
                    .foregroundColor(.secondary)
                
                Text(currentTier.displayName)
                    .font(.title3)
                    .fontWeight(.semibold)
            }
            
            Spacer()
            
            if currentTier == .studio {
                VStack(alignment: .trailing) {
                    Text("Premium")
                        .font(.caption)
                        .fontWeight(.semibold)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 4)
                        .background(.purple.opacity(0.2))
                        .foregroundColor(.purple)
                        .cornerRadius(8)
                    
                    Text("All features unlocked")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }
        }
        .padding()
        .background(.regularMaterial)
        .cornerRadius(12)
    }
    
    // MARK: - Subscription Tiers
    private var subscriptionTiersSection: some View {
        VStack(spacing: 16) {
            Text("Choose Your Plan")
                .font(.headline)
                .fontWeight(.semibold)
            
            LazyVStack(spacing: 12) {
                ForEach(SubscriptionTier.allCases, id: \.self) { tier in
                    SubscriptionTierCard(
                        tier: tier,
                        isSelected: selectedTier == tier,
                        isCurrent: currentTier == tier
                    ) {
                        selectedTier = tier
                    }
                }
            }
        }
    }
    
    // MARK: - Feature Comparison
    private var featureComparisonSection: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("What's Included")
                .font(.headline)
                .fontWeight(.semibold)
            
            VStack(spacing: 12) {
                FeatureRow(
                    title: "Basic Audio Mixing",
                    free: true,
                    pro: true,
                    studio: true
                )
                
                FeatureRow(
                    title: "Real-time Level Meters",
                    free: true,
                    pro: true,
                    studio: true
                )
                
                FeatureRow(
                    title: "Concurrent Streams",
                    free: "1",
                    pro: "4",
                    studio: "16"
                )
                
                FeatureRow(
                    title: "LUFS Metering",
                    free: false,
                    pro: true,
                    studio: true
                )
                
                FeatureRow(
                    title: "AI-Powered Processing",
                    free: false,
                    pro: true,
                    studio: true
                )
                
                FeatureRow(
                    title: "Custom LUFS Targeting",
                    free: false,
                    pro: true,
                    studio: true
                )
                
                FeatureRow(
                    title: "Advanced DSP Controls",
                    free: false,
                    pro: false,
                    studio: true
                )
                
                FeatureRow(
                    title: "Background Processing",
                    free: false,
                    pro: true,
                    studio: true
                )
                
                FeatureRow(
                    title: "Priority Support",
                    free: false,
                    pro: true,
                    studio: true
                )
                
                FeatureRow(
                    title: "Beta Features",
                    free: false,
                    pro: false,
                    studio: true
                )
            }
        }
        .padding()
        .background(.regularMaterial)
        .cornerRadius(12)
    }
    
    // MARK: - Upgrade Section
    private var upgradeSection: some View {
        VStack(spacing: 16) {
            Button(action: startPurchaseFlow) {
                VStack(spacing: 8) {
                    Text("Upgrade to \(selectedTier.displayName)")
                        .font(.headline)
                        .fontWeight(.semibold)
                    
                    Text(priceForTier(selectedTier))
                        .font(.subheadline)
                }
                .frame(maxWidth: .infinity)
                .padding(.vertical, 16)
                .background(.accentColor)
                .foregroundColor(.white)
                .cornerRadius(12)
            }
            .disabled(isProcessingPurchase || selectedTier <= currentTier)
            
            if isProcessingPurchase {
                HStack(spacing: 8) {
                    ProgressView()
                        .progressViewStyle(CircularProgressViewStyle())
                        .scaleEffect(0.8)
                    
                    Text("Processing purchase...")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }
            
            if currentTier == .free {
                Button("Try Pro Free for 7 Days") {
                    startFreeTrial()
                }
                .font(.subheadline)
                .fontWeight(.medium)
                .foregroundColor(.accentColor)
            }
        }
    }
    
    // MARK: - Legal Section
    private var legalSection: some View {
        VStack(spacing: 8) {
            Text("• Subscriptions auto-renew unless cancelled 24 hours before the end of the current period")
            Text("• Subscriptions can be managed and cancelled in App Store settings")
            Text("• Free trial automatically converts to paid subscription if not cancelled")
        }
        .font(.caption)
        .foregroundColor(.secondary)
        .multilineTextAlignment(.leading)
        .padding(.horizontal)
    }
    
    // MARK: - Helper Functions
    private func priceForTier(_ tier: SubscriptionTier) -> String {
        switch tier {
        case .free:
            return "Free"
        case .pro:
            return "$9.99/month"
        case .studio:
            return "$19.99/month"
        }
    }
    
    private func startPurchaseFlow() {
        showingPurchaseFlow = true
    }
    
    private func startFreeTrial() {
        isProcessingPurchase = true
        
        // Simulate purchase flow
        Task {
            try? await Task.sleep(for: .seconds(2))
            
            await MainActor.run {
                isProcessingPurchase = false
                // In real app, would start free trial
                dismiss()
            }
        }
    }
}

// MARK: - Subscription Tier Card
struct SubscriptionTierCard: View {
    let tier: SubscriptionTier
    let isSelected: Bool
    let isCurrent: Bool
    let onSelect: () -> Void
    
    var body: some View {
        Button(action: onSelect) {
            VStack(alignment: .leading, spacing: 12) {
                HStack {
                    VStack(alignment: .leading, spacing: 4) {
                        HStack(spacing: 8) {
                            Text(tier.displayName)
                                .font(.title3)
                                .fontWeight(.semibold)
                            
                            if isCurrent {
                                Text("CURRENT")
                                    .font(.caption2)
                                    .fontWeight(.bold)
                                    .padding(.horizontal, 6)
                                    .padding(.vertical, 2)
                                    .background(.green)
                                    .foregroundColor(.white)
                                    .cornerRadius(4)
                            }
                        }
                        
                        Text(priceText)
                            .font(.subheadline)
                            .foregroundColor(.secondary)
                    }
                    
                    Spacer()
                    
                    if isSelected && !isCurrent {
                        Image(systemName: "checkmark.circle.fill")
                            .font(.title2)
                            .foregroundColor(.accentColor)
                    } else if !isCurrent {
                        Circle()
                            .stroke(.quaternary, lineWidth: 2)
                            .frame(width: 24, height: 24)
                    }
                }
                
                Text(descriptionText)
                    .font(.caption)
                    .foregroundColor(.secondary)
                    .multilineTextAlignment(.leading)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            .padding()
            .background(backgroundContent)
            .cornerRadius(12)
        }
        .disabled(isCurrent)
        .buttonStyle(.plain)
    }
    
    private var priceText: String {
        switch tier {
        case .free: return "Free forever"
        case .pro: return "$9.99/month"
        case .studio: return "$19.99/month"
        }
    }
    
    private var descriptionText: String {
        switch tier {
        case .free:
            return "Perfect for getting started with basic audio mixing and real-time monitoring"
        case .pro:
            return "Advanced features for content creators and podcasters including AI processing and LUFS metering"
        case .studio:
            return "Professional-grade tools for audio engineers and producers with unlimited streams and priority support"
        }
    }
    
    @ViewBuilder
    private var backgroundContent: some View {
        if isCurrent {
            Color.green.opacity(0.1)
        } else if isSelected {
            Color.accentColor.opacity(0.1)
        } else {
            Color(.systemBackground)
        }
    }
}

// MARK: - Feature Row
struct FeatureRow: View {
    let title: String
    let free: Any
    let pro: Any
    let studio: Any
    
    var body: some View {
        VStack {
            HStack {
                Text(title)
                    .font(.subheadline)
                    .fontWeight(.medium)
                    .frame(maxWidth: .infinity, alignment: .leading)
                
                HStack(spacing: 0) {
                    FeatureCell(value: free, tier: .free)
                    FeatureCell(value: pro, tier: .pro)
                    FeatureCell(value: studio, tier: .studio)
                }
                .frame(width: 120)
            }
            
            Divider()
                .opacity(0.3)
        }
    }
}

struct FeatureCell: View {
    let value: Any
    let tier: SubscriptionTier
    
    var body: some View {
        Group {
            if let boolValue = value as? Bool {
                Image(systemName: boolValue ? "checkmark" : "xmark")
                    .font(.caption)
                    .foregroundColor(boolValue ? .green : .red)
                    .frame(width: 40, height: 20)
            } else if let stringValue = value as? String {
                Text(stringValue)
                    .font(.caption)
                    .fontWeight(.medium)
                    .frame(width: 40, height: 20)
            }
        }
    }
}

// MARK: - Purchase Flow View
struct PurchaseFlowView: View {
    let selectedTier: SubscriptionTier
    @Environment(\.dismiss) private var dismiss
    @State private var isProcessing = false
    
    var body: some View {
        NavigationView {
            VStack(spacing: 24) {
                Spacer()
                
                VStack(spacing: 16) {
                    Image(systemName: "creditcard")
                        .font(.system(size: 60))
                        .foregroundColor(.accentColor)
                    
                    Text("Complete Purchase")
                        .font(.title2)
                        .fontWeight(.bold)
                    
                    Text("Upgrade to \(selectedTier.displayName) for enhanced audio processing capabilities")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                        .multilineTextAlignment(.center)
                }
                
                Spacer()
                
                VStack(spacing: 16) {
                    Button("Purchase \(priceForTier(selectedTier))") {
                        processPurchase()
                    }
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(.accentColor)
                    .foregroundColor(.white)
                    .cornerRadius(12)
                    .disabled(isProcessing)
                    
                    if isProcessing {
                        ProgressView()
                            .progressViewStyle(CircularProgressViewStyle())
                    }
                }
            }
            .padding()
            .navigationTitle("Purchase")
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
    
    private func priceForTier(_ tier: SubscriptionTier) -> String {
        switch tier {
        case .free: return "Free"
        case .pro: return "$9.99/month"
        case .studio: return "$19.99/month"
        }
    }
    
    private func processPurchase() {
        isProcessing = true
        
        // Simulate purchase process
        Task {
            try? await Task.sleep(for: .seconds(3))
            
            await MainActor.run {
                isProcessing = false
                dismiss()
            }
        }
    }
}

#Preview {
    SubscriptionView()
        .environmentObject(ConfigurationManager())
}