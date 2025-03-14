```mermaid
gantt
    title GPU and CPU Reservations Timeline
    dateFormat  YYYY-MM-DD HH:mm
    section GPUs
    V100_2 :pending, 2025-03-15 19:00, 2025-03-22 19:00
    V100_3 :pending, 2025-03-22 19:05, 2025-03-29 19:05
    V100_4 :pending, 2025-03-29 20:15, 2025-04-05 20:15
    A100_pcie :pending, 2025-03-22 03:00, 2025-03-25 15:00
    A100_pcie_2 :pending, 2025-04-01 15:00, 2025-04-04 02:00
    V100_4_TACC :pending, 2025-04-05 20:20, 2025-04-12 20:20

    section CPUs
    cpu_server :active, 2025-03-14 01:30, 2025-03-21 01:30
