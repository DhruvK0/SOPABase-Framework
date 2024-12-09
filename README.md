# Multi-Agent Defense Copilot

This repository contains a multi-agent system implemented in Python, featuring three distinct agents that interact through a Flask web application. The system simulates strategic decision-making scenarios where defense agents compete and are evaluated by a judge.

## Project Structure

```
├── agents/
│   ├── coa.py         # Course of Action Agent
│   ├── adversary.py   # Adversary Agent
│   └── judge.py       # Judge Agent
│   └── eval.ipynb     # Eval Notebook for Coa Agent
│   └── eval_data.json # Eval Dataset
│   └── classifier.py  # Test Classifier
├── app.py             # Flask Application
└── README.md
```

## Components

### Course of Action Agent (`agents/coa.py`)
The Course of Action (COA) agent is responsible for:
- Generating strategic plans and actions
- Adapting to different scenarios
- Optimizing decisions based on available resources

### Adversary Agent (`agents/adversary.py`)
The Adversary agent:
- Analyzes and responds to proposed courses of action
- Implements counter-strategies
- Provides realistic opposition scenarios

### Judge Agent (`agents/judge.py`)
The Judge agent:
- Evaluates interactions between COA and Adversary agents
- Provides objective scoring and feedback
- Ensures fair play and rule compliance

### Flask Application (`app.py`)
The Flask application serves as the orchestration layer:
- Manages communication between agents
- Provides RESTful API endpoints
- Handles web interface for interaction and visualization

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

Dhruv Kanetkar - kanetkard@gmail.com