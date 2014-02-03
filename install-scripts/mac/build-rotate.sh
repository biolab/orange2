#!/bin/bash -e

function print_usage() {
    echo 'build-rotate.sh --match "PATTERN" [--remote [user@]host[:dir]] [--keep N] DIST_DIR

Delete old files matching the glob "PATTERN", in DIST_DIR keping only N
latest files.

NOTE: Always enclose PATTERN in quotes to prevent shell expansion at
      the call site.

NOTE: If --remote is specified then a sshfs executable must be on $PATH

Options:
    --match (-m) PATTERN    Glob pattern to match files in DIST_DIR
    --remote (-r) REMOTE    Remote sftp address. If supplied the remote
                            location will be mounted on DIST_DIR using sshfs
                            for the duration of the command.
    --keep (-k) N           Number of matching files to keep (default 10).

'
}

KEEP=10

while [[ ${1:0:1} = "-" ]]; do
    case $1 in
        -m|--match)
            MATCH="$2"
            shift 2
            ;;
        -k|--keep)
            KEEP="$2"
            shift 2
            ;;
        -r|--remote)
            REMOTE="$2"
            shift 2
            ;;
        -d|--dry-run)
            DRY_RUN=1
            shift 1
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            echo "Unknown option $1"
            print_usage
            exit 1
            ;;
    esac
done


DIST_DIR="$1"

if [[ ! $KEEP =~ ^[0-9]+$ ]]; then
    echo "Invalid --keep parameter $KEEP"
    exit 1
fi

if [[ ! $MATCH ]]; then
    echo "Invalid match parameter"
    print_usage
    exit 1
fi

UNMOUNT=

if [[ $REMOTE ]]; then
    if { ! mount | grep "$DIST_DIR" > /dev/null; } then
        echo "Mounting sshfs $REMOTE at $DIST_DIR"
        mkdir -p "$DIST_DIR"
        sshfs -o reconnect,uid=$(id -u),gid=$(id -g) \
            -o workaround=nonodelay:rename \
            "$REMOTE" \
            "$DIST_DIR"
        sleep 5
        UNMOUNT=1
    fi
fi

if [[ ! -d "$DIST_DIR" ]]; then
    echo "$DIST_DIR does not exist or is not a directory"
    exit 1
fi

FILES=( $(ls -t -1 "$DIST_DIR"/$MATCH 2> /dev/null || true) )

NFILES=${#FILES[*]}

echo "$NFILES files matching the pattern."
echo
echo "Removing old files"
echo "------------------"

for (( i=$KEEP; i < $NFILES; i++ )) do
    echo "removing ${FILES[$i]}"
    if [[ ! $DRY_RUN ]]; then
        rm "${FILES[$i]}"
    fi
done

sync

if [[ $UNMOUNT ]]; then
    diskutil unmount "$DIST_DIR"
fi
