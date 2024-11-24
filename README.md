# Separation Assurance in Urban Air Mobility Systems Using Shared Scheduling Protocols

This is the github repository that contains the testing environment for the paper: Separation Assurance in Urban Air Mobility Systems Using Shared Scheduling Protocols.

# Installation Ubunutu

## 1. Install project dependencies

1. Navigate to the UAM_shared_scheduling directory
    ```bash
    cd UAM_shared_scheduling
    ```
2. Install bluesky
    ```python
    pip install -e .
    ```

For more information on the BlueSky Simulator, please see: https://github.com/TUDelft-CNS-ATM/bluesky

# Config Parameters:

1. The simulation and model parameters are located in conf/config_test.gin.
2. If you would like to change the config file used when running main.py, modify the argument in line 383 of main.py:
    ```bash
    gin.parse_config_file("conf/config_test.gin")
    ```

## Key Parameters and Their Descriptions

The following parameters in the configuration file are important for controlling different aspects of the testing simulation:

- **Driver.scenario_file**  
  - Path to the scenario file used in the simulation. The number at the end of the scenario determines the number of aircraft per route.  
  - **Example:** `'scenarios/generated_scenarios/test_case_5.scn'`  

- **Driver.non_compliant_percentage**  
  - Percentage of non-compliant aircraft in the simulation.  
  - **Range:** 0 to 1 (e.g., 0.2 corresponds to 20% non-compliance).  
  - **Default:** `0` (all aircraft are compliant).  

- **Driver.protocol_active**  
  - Determines whether a coordination protocol is active during the simulation.  
  - **Type:** Boolean (`True` or `False`)  
  - **Default:** `True`.  

- **Driver.csma_cd_active**  
  - Enables or disables the CSMA/CD (Carrier-Sense Multiple Access with Collision Detection) protocol.  
  - **Type:** Boolean (`True` or `False`)  
  - **Default:** `True`.  

- **Driver.srtf_active**  
  - Enables or disables the Shortest Remaining Time First (SRTF) scheduling protocol.  
  - **Type:** Boolean (`True` or `False`)  
  - **Default:** `False`.  

- **Driver.round_robin_active**  
  - Enables or disables the Round Robin scheduling protocol.  
  - **Type:** Boolean (`True` or `False`)  
  - **Default:** `False`.  

- **Driver.testing_scenarios**  
  - Number of scenarios to test during the simulation.  
  - **Type:** Integer  
  - **Default:** `100`.  


# Running Project

1. Navigate to the UAM_shared_scheduling directory
    ```bash
    cd UAM_shared_scheduling
    ```
2. Try running main script
    ```python
    python main.py
    ````

# Visualization

1. Follow steps 1-2 above (Section: Running Project) in a single terminal (Terminal 1). Open a **second** terminal (Terminal 2) and follow the steps below

2. Navigate to the UAM_shared_scheduling directory
    ```bash
    cd UAM_shared_scheduling
    ```
3. Start BlueSky
    ```bash
    python BlueSky.py
    ```
4. The GUI should open up. After the GUI has started, in Terminal 1, run step 2 of **Running Project** to start the simulation.
5. In the BlueSky GUI, select the **Nodes** tab on the lower-right side. Select a different simulation node to see the Testing Environment sim.

# Acknowledgements:

This project builds on the code base created by Brittain et. al.: https://arxiv.org/pdf/2003.08353
