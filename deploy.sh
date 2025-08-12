#!/bin/bash

# Fast Docker Build Setup
# Run these commands once to speed up future builds

echo "Setting up Docker for faster builds..."

# 1. Enable BuildKit for faster builds
export DOCKER_BUILDKIT=1
export COMPOSE_DOCKER_CLI_BUILD=1

# 2. Create Docker daemon config for optimization
sudo mkdir -p /etc/docker
sudo tee /etc/docker/daemon.json > /dev/null <<EOF
{
  "features": {
    "buildkit": true
  },
  "builder": {
    "gc": {
      "enabled": true,
      "defaultKeepStorage": "20GB"
    }
  },
  "max-concurrent-downloads": 10,
  "max-concurrent-uploads": 10
}
EOF

# 3. Restart Docker to apply settings
sudo systemctl restart docker

# 4. Create buildx builder for advanced caching
docker buildx create --name graphrag-builder --use
docker buildx inspect --bootstrap

echo "Docker optimization complete!"

# Fast build commands
echo "Use these commands for fast builds:"
echo ""
echo "# Build with cache mount (fastest for rebuilds):"
echo "docker buildx build --cache-from type=local,src=/tmp/.buildx-cache --cache-to type=local,dest=/tmp/.buildx-cache -t graphrag-app ."
echo ""
echo "# Build with registry cache (for team sharing):"
echo "docker buildx build --cache-from type=registry,ref=your-registry/graphrag:cache --cache-to type=registry,ref=your-registry/graphrag:cache,mode=max -t graphrag-app ."
echo ""
echo "# Quick parallel build:"
echo "docker compose build --parallel --no-cache"