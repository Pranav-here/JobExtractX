```mermaid
gantt
    title GPU Reservations Timeline
    dateFormat  YYYY-MM-DD HH:mm
    section GPUs
    RTX 600  :active, 2025-02-15 13:00, 2025-02-17 01:00
    RTX 6000_2  :pending, 2025-02-17 01:00, 2025-02-24 01:00
    Train LLM 2 :pending, 2025-03-03 05:00, 2025-03-07 22:00
    Train LLM  :pending, 2025-03-06 00:00, 2025-03-07 00:00
    V100 :pending, 2025-03-07 19:00, 2025-03-10 11:00
    V100_2 :pending, 2025-03-15 19:00, 2025-03-22 19:00
    A100-pcie :pending, 2025-03-22 03:00, 2025-03-25 15:00
    A100-pcie :pending, 2025-04-01 15:00, 2025-04-04 02:00
