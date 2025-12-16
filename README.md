# hardsector_tool
A Tool to help Decode and reverse enineer Hard Sector Floppy Disk flux images
In an effort to build a tool and process for reverse engineering .scp Flux images of Hard Sectored Floppy Disks with the eventual aim of being able to identify their contents such as Directories anf Files.

./tests Contains Example .scp image files for testing
    ACMS80217 contains an .scp Flux Image file from a SS/DD NASHUA 8" Hard Sector 32 Floppy Disk which was created using the Greaseweazle card and tool set. The image creation command included the options --hardsector, --raw and --revs=5" to record 5 revolutions per track. The Disk is known to have 77 tracks (0-76).
        The Floppy disk image is thought to contain programs from the Wang OIS 100 Vintage computer System and may use one of the Wang style floppy formats such as 77 Tracks, FM encoded, 16 Hard Sectors per Track of 256 Data Bytes per Sector. This is what we need to initially confirm and the decode.
            The standard Hard Sector 32 Physical Disk hub includes 32 Sector holes and 1 Index Hole.
