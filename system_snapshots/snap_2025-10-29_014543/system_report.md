# System Snapshot (2025-10-29_014543)

**Repo:** /home/igc/igc
**Git branch:** main
**Git SHA:** f238011c200626421a0b7acf407447288e4388c0

## OS / Hardware

    ### /etc/os-release
    PRETTY_NAME="Ubuntu 24.04.3 LTS"
    NAME="Ubuntu"
    VERSION_ID="24.04"
    VERSION="24.04.3 LTS (Noble Numbat)"
    VERSION_CODENAME=noble
    ID=ubuntu
    ID_LIKE=debian
    HOME_URL="https://www.ubuntu.com/"
    SUPPORT_URL="https://help.ubuntu.com/"
    BUG_REPORT_URL="https://bugs.launchpad.net/ubuntu/"
    PRIVACY_POLICY_URL="https://www.ubuntu.com/legal/terms-and-policies/privacy-policy"
    UBUNTU_CODENAME=noble
    LOGO=ubuntu-logo
    
    ### Kernel
    Linux igcapp 6.8.0-86-generic #87-Ubuntu SMP PREEMPT_DYNAMIC Mon Sep 22 18:03:36 UTC 2025 x86_64 x86_64 x86_64 GNU/Linux
    
    ### CPU
    /usr/bin/lscpu
    Architecture:                         x86_64
    CPU op-mode(s):                       32-bit, 64-bit
    Address sizes:                        43 bits physical, 48 bits virtual
    Byte Order:                           Little Endian
    CPU(s):                               12
    On-line CPU(s) list:                  0-11
    Vendor ID:                            AuthenticAMD
    Model name:                           AMD Ryzen 5 3600 6-Core Processor
    CPU family:                           23
    Model:                                113
    Thread(s) per core:                   2
    Core(s) per socket:                   6
    Socket(s):                            1
    Stepping:                             0
    Frequency boost:                      enabled
    CPU(s) scaling MHz:                   72%
    CPU max MHz:                          3600.0000
    CPU min MHz:                          2200.0000
    BogoMIPS:                             7199.83
    Flags:                                fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht syscall nx mmxext fxsr_opt pdpe1gb rdtscp lm constant_tsc rep_good nopl nonstop_tsc cpuid extd_apicid aperfmperf rapl pni pclmulqdq monitor ssse3 fma cx16 sse4_1 sse4_2 x2apic movbe popcnt aes xsave avx f16c rdrand lahf_lm cmp_legacy svm extapic cr8_legacy abm sse4a misalignsse 3dnowprefetch osvw ibs skinit wdt tce topoext perfctr_core perfctr_nb bpext perfctr_llc mwaitx cpb cat_l3 cdp_l3 hw_pstate ssbd mba ibpb stibp vmmcall fsgsbase bmi1 avx2 smep bmi2 cqm rdt_a rdseed adx smap clflushopt clwb sha_ni xsaveopt xsavec xgetbv1 cqm_llc cqm_occup_llc cqm_mbm_total cqm_mbm_local clzero irperf xsaveerptr rdpru wbnoinvd arat npt lbrv svm_lock nrip_save tsc_scale vmcb_clean flushbyasid decodeassists pausefilter pfthreshold avic v_vmsave_vmload vgif v_spec_ctrl umip rdpid overflow_recov succor smca sev sev_es
    Virtualization:                       AMD-V
    L1d cache:                            192 KiB (6 instances)
    L1i cache:                            192 KiB (6 instances)
    L2 cache:                             3 MiB (6 instances)
    L3 cache:                             32 MiB (2 instances)
    NUMA node(s):                         1
    NUMA node0 CPU(s):                    0-11
    Vulnerability Gather data sampling:   Not affected
    Vulnerability Itlb multihit:          Not affected
    Vulnerability L1tf:                   Not affected
    Vulnerability Mds:                    Not affected
    Vulnerability Meltdown:               Not affected
    Vulnerability Mmio stale data:        Not affected
    Vulnerability Reg file data sampling: Not affected
    Vulnerability Retbleed:               Mitigation; untrained return thunk; SMT enabled with STIBP protection
    Vulnerability Spec rstack overflow:   Mitigation; Safe RET
    Vulnerability Spec store bypass:      Mitigation; Speculative Store Bypass disabled via prctl
    Vulnerability Spectre v1:             Mitigation; usercopy/swapgs barriers and __user pointer sanitization
    Vulnerability Spectre v2:             Mitigation; Retpolines; IBPB conditional; STIBP always-on; RSB filling; PBRSB-eIBRS Not affected; BHI Not affected
    Vulnerability Srbds:                  Not affected
    Vulnerability Tsx async abort:        Not affected
    
    ### Memory
    /usr/bin/free
                   total        used        free      shared  buff/cache   available
    Mem:            62Gi       1.6Gi        57Gi        31Mi       4.6Gi        61Gi
    Swap:          8.0Gi          0B       8.0Gi
    
    ### Disk (.)
    Filesystem      Size  Used Avail Use% Mounted on
    /dev/nvme0n1p3  460G  5.3G  431G   2% /

## Python / venv

**VIRTUAL_ENV:** /home/igc/igc/.venv
**python:** /home/igc/igc/.venv/bin/python3
**pip:** /home/igc/igc/.venv/bin/pip
    ### Versions
    Python 3.12.3
    pip 25.3 from /home/igc/igc/.venv/lib/python3.12/site-packages/pip (python 3.12)

## pip packages

- Saved **pip list** → `system_snapshots/snap_2025-10-29_014543/pip_list.txt`
- Saved **pip freeze** (lock) → `system_snapshots/snap_2025-10-29_014543/requirements-lock.txt`
- pipdeptree not installed (optional). Install: `pip install pipdeptree`

## Key Python libraries (NumPy/SciPy/pyFFTW/GUDHI/FastAPI/uvicorn)

- Saved **python libs JSON** → `system_snapshots/snap_2025-10-29_014543/python_libs.json`

## Python libs summary (excerpt)

