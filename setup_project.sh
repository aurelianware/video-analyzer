#!/bin/bash

echo "🚀 Setting up VS Code project..."

# Create .vscode folder and settings.json
mkdir -p .vscode
cat > .vscode/settings.json <<EOL
{
    "git.autofetch": true,
    "git.confirmSync": false,
    "git.enableSmartCommit": true,
    "git.inputValidation": "warn",
    "gitlens.hovers.enabled": true,
    "gitlens.currentLine.enabled": true,
    "gitlens.codeLens.enabled": true,
    "githubPullRequests.telemetry.enabled": false,
    "github.copilot.enable": {
        "*": true,
        "plaintext": false
    }
}
EOL

echo "✅ .vscode/settings.json created"

# Create .gitignore
cat > .gitignore <<EOL
# Python
__pycache__/
*.pyc
*.pyo
*.pyd
.env/
.venv/
myenv/

# VS Code
.vscode/

# Video & output files
*.mp4
*.avi
*.mov
cropped_*/
*.png
*.jpg
EOL

echo "✅ .gitignore created"

# Initialize git repo
git init
git add .
git commit -m "Initial project setup with VS Code settings and gitignore"

echo "✅ Git repository initialized and first commit made"

echo "🎉 Setup complete! Next steps:"
echo "👉 Run: gh repo create video-analyzer --public"
echo "👉 Run: git push -u origin main"

    