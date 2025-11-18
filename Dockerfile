# 1. Start from the official, clean Node.js 20 image (Debian-based)
FROM node:20

# 2. Add Python 3 and vim
RUN apt-get update && apt-get install -y python3 python3-pip vim

# 3. Install the latest Gemini CLI globally inside the container
RUN npm install -g @google/gemini-cli --no-update-notifier

# 4. Verify the install for our build logs
RUN node -v
RUN python3 --version
RUN gemini --version

# 5. Set a working directory (good practice)
WORKDIR /app

# 6. Set the default command. When the container starts,
#    it will just run "bash", giving you an interactive shell.
CMD ["/bin/bash"]
