from datasets import load_dataset

ds = load_dataset("emozilla/yarn-train-tokenized-16k-mistral")

'''
ds = load_dataset("togethercomputer/RedPajama-Data-V2",
                  name="default",
                  partition="head_middle",
                  snapshots=["2023-14"],
                  languages=["en"],
                  cache_dir="/mnt/esperanto/et/Adrien/context_extension/dataset",
                  data_dir='/mnt/esperanto/et/Adrien/context_extension/dataset',
                  )


Filesystem      Size  Used Avail Use% Mounted on
udev            236G     0  236G   0% /dev
tmpfs            51G  3.3M   51G   1% /run
/dev/nvme0n1p2  879G  616G  219G  74% /
tmpfs           252G  296K  252G   1% /dev/shm
tmpfs           5.0M     0  5.0M   0% /run/lock
tmpfs           252G     0  252G   0% /sys/fs/cgroup
/dev/loop2       74M   74M     0 100% /snap/core22/858
/dev/loop0      128K  128K     0 100% /snap/bare/5
/dev/loop3       74M   74M     0 100% /snap/core22/864
/dev/nvme0n1p1  511M  6.1M  505M   2% /boot/efi
/dev/loop6      350M  350M     0 100% /snap/gnome-3-38-2004/143
/dev/loop5       64M   64M     0 100% /snap/core20/1974
/dev/loop7      497M  497M     0 100% /snap/gnome-42-2204/132
/dev/loop8       92M   92M     0 100% /snap/gtk-common-themes/1535
/dev/loop9      350M  350M     0 100% /snap/gnome-3-38-2004/140
/dev/loop10     497M  497M     0 100% /snap/gnome-42-2204/141
/dev/loop11      67M   67M     0 100% /snap/cups/962
/dev/nvme1n1    7.0T  398G  6.2T   6% /mnt/esperanto
/dev/loop12      13M   13M     0 100% /snap/snap-store/959
/dev/loop13      41M   41M     0 100% /snap/snapd/20290
/dev/loop14      66M   66M     0 100% /snap/gtk-common-themes/1519
/dev/loop15      64M   64M     0 100% /snap/core20/2015
/dev/loop17      46M   46M     0 100% /snap/snap-store/638
/dev/loop16      67M   67M     0 100% /snap/cups/980
/dev/loop18      41M   41M     0 100% /snap/snapd/20092
tmpfs            51G   20K   51G   1% /run/user/125
/dev/loop19     158M  158M     0 100% /snap/chromium/2686
tmpfs            51G  4.0K   51G   1% /run/user/1000
/dev/loop1      158M  158M     0 100% /snap/chromium/2695
'''