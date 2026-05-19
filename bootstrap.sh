#!/usr/bin/env bash
set -euo pipefail

MARKER="/workspace/.vast_bootstrap_done"

if [[ -f "$MARKER" ]]; then
    echo "Vast bootstrap already ran. Delete $MARKER to rerun."
    exit 0
fi

export DEBIAN_FRONTEND=noninteractive

echo "==> Updating apt"
apt-get update

echo "==> Installing missing dev tools"
apt-get install -y \
    just \
    clang \
    clangd \
    clang-format \
    lldb \
    gh \
    vim \
    tmux \
    htop \
    ripgrep \
    fd-find \
    tree \
    jq \
    unzip \
    zip \
    ca-certificates \
    curl \
    wget \
    build-essential \
    pkg-config

echo "==> Configuring git"
git config --global user.name Aaryan Rampal
git config --global user.email 76913929+aaryan-rampal@users.noreply.github.com
git config --global init.defaultBranch main
git config --global pull.rebase false
git config --global core.editor "vim"
git config --global color.ui auto

echo "==> Setting up persistent SSH key"
mkdir -p /workspace/.ssh
chmod 700 /workspace/.ssh

if [[ ! -f /workspace/.ssh/id_ed25519 ]]; then
    ssh-keygen -t ed25519 -N "" \
        -f /workspace/.ssh/id_ed25519 \
        -C "${GIT_EMAIL:-vast-ai-devbox}"
fi

mkdir -p /root/.ssh
chmod 700 /root/.ssh

ln -sf /workspace/.ssh/id_ed25519 /root/.ssh/id_ed25519
ln -sf /workspace/.ssh/id_ed25519.pub /root/.ssh/id_ed25519.pub

cat > /root/.ssh/config <<'EOF'
Host github.com
  HostName github.com
  User git
  IdentityFile /root/.ssh/id_ed25519
  StrictHostKeyChecking accept-new
EOF

chmod 600 /root/.ssh/config

echo "==> Adding shell defaults"
cat >> /root/.bashrc <<'EOF'

# CUDA dev defaults
alias ll='ls -lah'
alias gs='git status'
alias ga='git add'
alias gc='git commit'
alias gp='git push'
alias nv='nvidia-smi'
alias j='just'

cd /workspace
EOF

echo "==> Tool versions"
echo "--- CUDA ---"
nvcc --version || true
nvidia-smi || true

echo "--- Profiling ---"
ncu --version || true
compute-sanitizer --version || true
cuobjdump --version || true
nvdisasm --version || true

echo "--- Dev tools ---"
gcc --version | head -1 || true
g++ --version | head -1 || true
clang --version | head -1 || true
clangd --version | head -1 || true
clang-format --version || true
just --version || true
gh --version | head -1 || true
git --version || true

echo
echo "==> GitHub SSH public key:"
cat /workspace/.ssh/id_ed25519.pub
echo
echo "Add this key to GitHub if not already added."

touch "$MARKER"

echo "==> Bootstrap complete."
