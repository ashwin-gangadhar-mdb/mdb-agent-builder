#!/bin/bash
# verify_installation.sh - Script to verify the installation and configuration of MDB Agent Builder

# Color codes for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo "===== MDB Agent Builder Installation Verification ====="
echo

# Check for Python installation
echo -n "Checking Python installation... "
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    echo -e "${GREEN}Found $PYTHON_VERSION${NC}"
else
    echo -e "${RED}Python 3 not found${NC}"
    echo "Please install Python 3.8 or higher"
    exit 1
fi

# Check for virtual environment
echo -n "Checking for virtual environment... "
if [ -d ".venv" ]; then
    echo -e "${GREEN}Found .venv directory${NC}"
else
    echo -e "${YELLOW}Not found${NC}"
    echo "Run 'make setup-env' to create a virtual environment"
fi

# Check for package installation
echo -n "Checking for package installation... "
if [ -d "agent_builder.egg-info" ]; then
    echo -e "${GREEN}Package installed in development mode${NC}"
else
    echo -e "${YELLOW}Not installed${NC}"
    echo "Run 'make install' to install the package"
fi

# Check for configuration files
echo -n "Checking for configuration files... "
if [ -d "config" ] && [ -f "config/agents.yaml" ]; then
    echo -e "${GREEN}Configuration files found${NC}"
else
    echo -e "${YELLOW}Missing configuration${NC}"
    echo "Run 'make create-config' to create configuration files"
fi

# Check for .env file
echo -n "Checking for .env file... "
if [ -f ".env" ]; then
    echo -e "${GREEN}Found .env file${NC}"
else
    echo -e "${YELLOW}Not found${NC}"
    echo "Copy .env.example to .env and edit with your settings"
fi

# Check for required directories
echo -n "Checking for required directories... "
MISSING_DIRS=""
for dir in "config" "logs" "prompts"; do
    if [ ! -d "$dir" ]; then
        MISSING_DIRS="$MISSING_DIRS $dir"
    fi
done

if [ -z "$MISSING_DIRS" ]; then
    echo -e "${GREEN}All required directories exist${NC}"
else
    echo -e "${YELLOW}Missing:$MISSING_DIRS${NC}"
    echo "Run 'make create-config' to create required directories"
fi

echo
echo "===== Verification Complete ====="
echo
echo "To complete setup:"
echo "1. Activate virtual environment: source .venv/bin/activate"
echo "2. Install package: make install"
echo "3. Create configuration: make create-config"
echo "4. Copy .env.example to .env and edit settings"
echo "5. Run the server: make run"
echo
echo "For more options, run: make help"
