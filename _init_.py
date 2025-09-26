sai-integrated-system/
├── .vscode/
├── models/
│   └── yolov11_model.pt
├── core/
│   ├── __init__.py
│   ├── fitness_trainer.py  # Our existing code
│   ├── sai_assessor.py     # SAI assessment logic
│   └── performance_evaluator.py
├── utils/
│   ├── pose_analysis.py
│   └── visualization.py
├── requirements.txt
└── main.py                 # Integrated main application