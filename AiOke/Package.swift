// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "AiOke",
    platforms: [
        .iOS(.v15),
        .macOS(.v12)
    ],
    products: [
        .library(name: "AiOke", targets: ["AiOke"]),
        .executable(name: "AiOkeApp", targets: ["AiOkeApp"])
    ],
    dependencies: [
        // No external dependencies for MVP - using only Apple frameworks
    ],
    targets: [
        .target(
            name: "AiOke",
            dependencies: [],
            path: "Sources/AiOke"
        ),
        .executableTarget(
            name: "AiOkeApp",
            dependencies: ["AiOke"],
            path: "Sources/AiOkeApp"
        ),
        .testTarget(
            name: "AiOkeTests",
            dependencies: ["AiOke"],
            path: "Tests/AiOkeTests"
        )
    ]
)