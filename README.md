# AgriArm

A robotic arm that finds and picks ripe strawberries using a camera and AI. The perception side handles detection and ripeness classification. The hardware side handles the arm, gripper and motion planning.

# Branch structure
```
main                        ← always working
├── software-dev            ← integration branch for software
│   └── software-feature/*  ← one branch per feature or task
└── hardware-dev            ← integration branch for hardware
    └── hardware-feature/*  ← one branch per feature or task
```
fix/description             ← bug fixes, branch off whichever dev is relevant

Never commit directly to main. When you start new work, it goes on a feature branch, gets merged into the relevant dev branch first, then into main (Aaron will handle PR accepting).

# Workflow
1. Pull (please) the latest from your dev branch before starting anything
2. Create a feature branch off it, software-feature/your-task or hardware-feature/your-task
3. Commit regularly with a short message describing what changed
4. Open a PR into the dev branch, gets reviewed by Aaron or Isaac, then merge
5. Delete the branch after merging

For bugs: branch off fix/short-description, fix it and make PR


# Contacts
Perception / software — Aaron: aliu0064@student.monash.edu
Hardware / kinematics — Isaac

# Getting started
```
conda create -n agriarm python=3.12 -y
conda activate agriarm
pip install -r requirements.txt
```
