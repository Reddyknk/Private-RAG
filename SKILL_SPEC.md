# Specification for Creating Skills for the Agent

This document details the modular layout, schemas, configurations, and core python runner infrastructure needed to build an autonomous agent containing domain-specific capabilities using the official google-genai SDK.

## 📂 Directory Architecture
The layout isolates domain procedural logic into standard structural boundaries:
```
my_agent/
├── app.py                      # Core Orchestrator utilizing google-genai SDK
├── requirements.txt            # Package declarations
└── skills/
    ├── myskill-1-skill/
    │   ├── SKILL.md            # Metadata & SOP for Skill 1
    │   └── scripts/
    │       └── env_tools.py    # Public Open-Meteo & Time logic (No Keys Req.)
    ├── myskill-2-skill/
    │   ├── SKILL.md            # Metadata & SOP for Skill 2
    │   └── data/
    │       └── registry.csv    # Flat-file database for record resolution
    └── myskill-3-skill/
        ├── SKILL.md            # Metadata & SOP for Skill 3
        └── templates/
            └── myskill_info.md   # Plant care instructions template
```
## 🧱 Agent Execution Flow
When the Agent discovers a new skill, it will use ollama to embed the skill.md file and store the embedding in the vector database. The embedding will be used to retrieve the skill.md file when the user asks for information that might be in the skill. The agent will then use the skill.md file to answer the user's question.