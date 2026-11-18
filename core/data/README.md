# Audit syscall number data

`linux_syscalls_v6_12.json` records only syscall number-to-name ABI facts from Linux v6.12. No kernel code is shipped. The source tables are tagged `GPL-2.0 WITH Linux-syscall-note`:

- [`arch/x86/entry/syscalls/syscall_64.tbl`](https://github.com/torvalds/linux/blob/v6.12/arch/x86/entry/syscalls/syscall_64.tbl): `common,64`.
- [`arch/arm64/tools/syscall_64.tbl`](https://github.com/torvalds/linux/blob/v6.12/arch/arm64/tools/syscall_64.tbl): `common,64,renameat,rlimit,memfd_secret`, as selected in [`arch/arm64/kernel/Makefile.syscalls`](https://github.com/torvalds/linux/blob/v6.12/arch/arm64/kernel/Makefile.syscalls).

Native audit architecture IDs are `c000003e` and `c00000b7`. Other ABIs and newer syscall numbers are passed through as architecture-qualified numeric tokens rather than guessed from the analysis host.
