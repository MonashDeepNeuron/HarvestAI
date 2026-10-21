# HarvestAI

A robotic arm that finds and picks ripe strawberries using a camera and AI. The perception side handles detection and ripeness classification. The hardware side handles the arm, gripper and motion planning.

Built on **ROS 2 Jazzy** (Ubuntu 24.04), with Python for almost everything.

## Contents

- [Repo layout](#repo-layout)
- [Getting started (first time)](#getting-started-first-time)
  - [1. Install WSL and Ubuntu 24.04](#1-install-wsl-and-ubuntu-2404)
  - [2. Open the Ubuntu terminal](#2-open-the-ubuntu-terminal)
  - [3. Check and update](#3-check-and-update)
  - [4. GPU check (NVIDIA only)](#4-gpu-check-nvidia-only)
  - [5. Install ROS 2 Jazzy](#5-install-ros-2-jazzy)
  - [6. Install uv and Git](#6-install-uv-and-git)
  - [7. Connect to GitHub with an SSH key](#7-connect-to-github-with-an-ssh-key)
  - [8. Clone the repo into a workspace](#8-clone-the-repo-into-a-workspace)
  - [9. Create the Python environment](#9-create-the-python-environment)
  - [10. Build the workspace](#10-build-the-workspace)
  - [11. Set up VS Code](#11-set-up-vs-code)
- [Every time you work](#every-time-you-work)
- [Python dependencies](#python-dependencies)
- [Branch structure](#branch-structure)
- [Workflow](#workflow)
- [Troubleshooting](#troubleshooting)
- [Contacts](#contacts)

---

## Repo layout

Each `harvestai_*` folder is a ROS 2 package with one job. Packages talk to each other through ROS messages, never by importing each other's code.

```
HarvestAI/
├── harvestai_interfaces/    message formats shared by every package (msg/BerryDetection.msg)
├── harvestai_description/   robot model: URDF, meshes, RViz config
├── harvestai_bringup/       launch files and config that start the system (sim or real)
├── harvestai_perception/    camera in, strawberry detections out
│   └── harvestai_perception/
│       ├── models/          trained YOLO and ripeness weights (.pt)
│       ├── notebooks/
│       ├── ripeness/        ripeness classifier: train.py, evaluate.py
│       └── ui/              Gradio app for trying the models
├── harvestai_tasks/         task logic, kinematics and planning
├── docs/
├── dependencies.repos       ROS packages cloned from source (vcstool)
├── requirements.in          Python libraries: edit this one
└── requirements.txt         pinned versions, generated from requirements.in (don't edit)
```

> **Why two `harvestai_perception` folders?** The outer one is the ROS package (packaging files). The inner one is the Python module, and your code goes there.

---

## Getting started (first time)

You need **Windows 11** (or Windows 10 22H2+) with admin rights, about **40 GB free**, and virtualisation enabled (Task Manager → Performance → CPU → *Virtualization: Enabled*).
Already on Ubuntu 24.04? Skip to [step 5](#5-install-ros-2-jazzy). On a Mac, use an Ubuntu 24.04 VM.

### 1. Install WSL and Ubuntu 24.04

In **PowerShell as administrator**:

```powershell
wsl --install -d Ubuntu-24.04
```

Restart when asked.

### 2. Open the Ubuntu terminal

Any of these:
- Start menu → type **Ubuntu** → **Ubuntu 24.04**
- Windows Terminal → the **⌄** next to **+** → **Ubuntu-24.04**
- PowerShell → type `wsl`

The first launch asks you to create a **Linux username and password** (separate from Windows). Nothing shows while you type the password. That's normal.

You're in Ubuntu when the prompt looks like `you@PC:~$`, not `PS C:\...>`.

### 3. Check and update

In PowerShell:
```powershell
wsl -l -v          # Ubuntu-24.04 should show VERSION 2
wsl --update
```
In Ubuntu:
```bash
sudo apt update && sudo apt upgrade -y
```

### 4. GPU check (NVIDIA only)

Install the NVIDIA driver **on Windows only**. Never install one inside Ubuntu. Then in Ubuntu:
```bash
nvidia-smi         # should list your GPU
```

### 5. Install ROS 2 Jazzy

From the [official guide](https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html). If a line fails, copy that step from the guide.

```bash
sudo apt install -y software-properties-common curl
sudo add-apt-repository -y universe

V=$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest \
  | grep -F "tag_name" | awk -F\" '{print $4}')
BASE=https://github.com/ros-infrastructure/ros-apt-source/releases/download
CODENAME=$(. /etc/os-release && echo $VERSION_CODENAME)
curl -L -o /tmp/ros2-apt-source.deb "$BASE/$V/ros2-apt-source_$V.${CODENAME}_all.deb"
sudo dpkg -i /tmp/ros2-apt-source.deb

sudo apt update
sudo apt install -y ros-jazzy-desktop ros-dev-tools

# load ROS in every new terminal (don't skip this)
echo "source /opt/ros/jazzy/setup.bash" >> ~/.bashrc
source ~/.bashrc

# one-time rosdep setup
sudo rosdep init
rosdep update
```

Check: `echo $ROS_DISTRO` prints `jazzy`.

### 6. Install uv and Git

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

sudo apt install -y git
git config --global user.name "Your Name"
git config --global user.email "you@student.monash.edu"   # same email as your GitHub account
```

### 7. Connect to GitHub with an SSH key

This lets you `git push` without typing a password or token.

```bash
ssh-keygen -t ed25519 -C "you@student.monash.edu"   # press Enter at every prompt
cat ~/.ssh/id_ed25519.pub | clip.exe                # copies the key to your Windows clipboard
```

On GitHub: **Settings → SSH and GPG keys → New SSH key**. Give it a title (e.g. *My PC – WSL*), then paste into **Key**. It must be **one line starting with `ssh-ed25519`**. Paste the key itself, not the `cat` command.

```bash
ssh -T git@github.com
# "Hi <name>! You've successfully authenticated, but GitHub does not provide shell access."  ← success
```

> Only ever share the file ending in **`.pub`**.

### 8. Clone the repo into a workspace

Keep code on the **Linux side** (`~/...`), never under `/mnt/c/...`, which is many times slower. Check with `pwd`.

```bash
mkdir -p ~/harvest_ws/src && cd ~/harvest_ws/src
git clone git@github.com:MonashDeepNeuron/HarvestAI.git
cd HarvestAI
```

### 9. Create the Python environment

```bash
uv venv --python /usr/bin/python3 --system-site-packages .venv
touch .venv/COLCON_IGNORE
source .venv/bin/activate
uv pip sync requirements.txt
python -c "import rclpy, ultralytics, gradio; print('ok')"
```

| Flag | Why |
|---|---|
| `--python /usr/bin/python3` | ROS 2 only works with Ubuntu's own Python 3.12. **Don't use `uv python install`.** |
| `--system-site-packages` | Lets the venv see system-installed packages as well as your pip ones |
| `COLCON_IGNORE` | Stops the ROS build from trying to build the venv |

### 10. Build the workspace

Always build from `~/harvest_ws`, never from inside the repo.

```bash
cd ~/harvest_ws
rosdep install --from-paths src --ignore-src -y
colcon build --symlink-install
source install/setup.bash

# checks
ros2 interface show harvestai_interfaces/msg/BerryDetection
ros2 run harvestai_perception perception_node
```

### 11. Set up VS Code

1. Install [VS Code](https://code.visualstudio.com) on Windows, plus the **WSL** extension.
2. From Ubuntu: `cd ~/harvest_ws/src/HarvestAI && code .`
3. The bottom-left corner should say **WSL: Ubuntu-24.04**. Install the **Python** extension again there.
4. `Ctrl+Shift+P` → *Python: Select Interpreter* → `.venv/bin/python`.

Edit files through VS Code, not Windows File Explorer (`\\wsl$\...`), which can add stray files and Windows line endings.

---

## Every time you work

```bash
cd ~/harvest_ws/src/HarvestAI
source .venv/bin/activate
git checkout <your-branch> && git pull
uv pip sync requirements.txt          # only if requirements.txt changed
cd ~/harvest_ws && colcon build --symlink-install && source install/setup.bash
```

Run the perception UI:
```bash
cd ~/harvest_ws/src/HarvestAI/harvestai_perception/harvestai_perception/ui
python app.py                         # open the http://127.0.0.1:7860 link
```

---

## Python dependencies

| File | What it is | Edit it? |
|---|---|---|
| `requirements.in` | The libraries we use, loosely versioned | ✅ Yes |
| `requirements.txt` | Every library pinned to an exact version, generated | ❌ Never by hand |

**Adding a library:**
```bash
# 1. add a line to requirements.in, e.g.  onnxruntime>=1.20
uv pip compile --universal requirements.in -o requirements.txt
uv pip sync requirements.txt
git add requirements.in requirements.txt
git commit -m "Add onnxruntime"
```

**After pulling someone else's change:** `uv pip sync requirements.txt`

Don't `uv pip install` one-off libraries into `.venv`, because the next `sync` removes them. ROS packages go in a package's `package.xml` (installed by `rosdep`), not in `requirements.in`.

---

## Branch structure

We use **GitHub Flow**: one long-lived branch (`main`) and short-lived branches for each task.

```
main                          ← always builds; protected, changes only arrive by pull request
├── software/<task>           ← e.g. software/ripeness-node
├── hardware/<task>           ← e.g. hardware/gripper-urdf
└── fix/<short-description>   ← e.g. fix/detector-model-path
```

- **Never commit directly to `main`.** GitHub blocks it, and every change goes through a pull request (PR).
- **Branch names:** `software/`, `hardware/` or `fix/`, then a few words in kebab-case.
- **Keep branches short-lived:** days, not weeks. Smaller PRs are quicker to review and rarely conflict.
- **Stable versions for demos** are git tags (e.g. `v0.1-demo`), not branches.

## Workflow

1. **Start from the latest `main`**
    ```bash
    git checkout main && git pull
    git checkout -b software/your-task
    ```
2. **Work and commit regularly**, with a short message saying what changed
    ```bash
    git add .
    git commit -m "Add ripeness classifier node"
    ```
3. **Push and open a PR into `main`**
    ```bash
    git push -u origin software/your-task
    ```
    GitHub prints a link to open the PR. Ask for a review from Aaron (software) or Isaac (hardware).
4. **Keep your branch up to date** if `main` moves on while you're working
    ```bash
    git checkout main && git pull
    git checkout software/your-task && git merge main
    ```
5. **Merge once it's approved and builds**, then delete the branch (GitHub offers a button). Locally:
    ```bash
    git checkout main && git pull
    git branch -d software/your-task
    ```

**Which branch am I on?** Run `git branch` (the one marked `*`), or look at VS Code's bottom-left corner.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `No module named 'rclpy'` | ROS not loaded: `source /opt/ros/jazzy/setup.bash`. If `python --version` isn't 3.12, rebuild `.venv` (step 9). |
| `No module named 'gradio'` / `ultralytics` | `source .venv/bin/activate`, then `uv pip sync requirements.txt` |
| `rosdep installation has not been initialized` | `sudo rosdep init && rosdep update` |
| `Package 'harvestai_...' not found` | `source ~/harvest_ws/install/setup.bash` (after every build, in every terminal) |
| A message file "doesn't exist" during build | Message files must be CamelCase (`BerryDetection.msg`, not `berry_detection.msg`) and listed in `CMakeLists.txt` |
| GitHub rejects the SSH key | Paste the output of `cat ~/.ssh/id_ed25519.pub`, starting with `ssh-ed25519` |
| Everything is very slow | The repo is under `/mnt/c`. Re-clone into `~/harvest_ws/src` |
| `build/`, `install/`, `log/` show up in git | You ran `colcon build` inside the repo. Delete them and build from `~/harvest_ws` |
| `nvidia-smi` not found | Install the NVIDIA driver on Windows, never inside Ubuntu |

---

## Contacts
Perception / software — Aaron: aliu0064@student.monash.edu
Hardware / kinematics — Isaac
