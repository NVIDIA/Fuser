# start from the official, clean Node.js 20 image (Debian-based)
#  Gemini seems to go haywire with older node versions (e.g. 18)
FROM node:20

COPY pr_preflight_launcher.py /usr/local/bin/
COPY utils.py /usr/local/bin/tools/
COPY git_helpers.py /usr/local/bin/tools/
COPY ai_cli_wrapper.py /usr/local/bin/tools/

# Add Python 3 and vim
RUN apt-get update && apt-get install -y python3 python3-pip vim

# Install the latest Gemini CLI globally inside the container
RUN npm install -g @google/gemini-cli --no-update-notifier

RUN node -v
RUN python3 --version
RUN gemini --version

# 5. Set a working directory (good practice)
WORKDIR /app

# 6. Set the default command. When the container starts,
#    it will just run "bash", giving you an interactive shell.
CMD ["/bin/bash"]
