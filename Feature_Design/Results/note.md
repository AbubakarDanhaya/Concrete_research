                    911 observations
                          │
                 80/20 train-test split
                          │
             ┌────────────┴────────────┐
             │                         │
        728 Training                183 Test
             │                         │
       5-fold CV                    LOCKED
             │                         │
   Model selection/tuning              │
             │                         │
             └────────────┐            │
                          ▼            │
                    Final model        │
                          │            │
                          └────────────┘
                               │
                               ▼
                       Final test metrics