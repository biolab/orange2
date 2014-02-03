#!/bin/bash -e

function print_usage() {
    echo 'build-publish.sh [--remote [user@]host[:dir]] --key KEY -d DIST_DIR FILE

Publish a build FILE in DIST_DIR under KEY

NOTE: KEY  needs to match [A-Z_]+
NOTE: If --remote is specified then a sshfs executable must be on $PATH

Options:
    --remote (-r) REMOTE  A "sshfs host" string. If present then the remote
                          will be mounted under DIST_DIR.
    --key (-k)            Key under which to publish the file.
    --dist-dir (-d)       Directory where to publish.
'
}

# section "Section title" "delimiter"
# Note: Can't use "-" as the delimiter
function section() {
    line="$1"
    delim="$2"
    echo "$line"
    len=${#line}
    printf "${delim}%.0s" $(eval "echo {1.."$len"}")
    echo

}

while [ ${1:0:1} = "-" ];  do
    case $1 in
        --key)
            KEY=$2
            shift 2
            ;;
        -r|--remote)
            REMOTE=$2
            shift 2
            ;;
        -d|--dist-dir)
            DIST_DIR=$2
            shift 2
            ;;
        *)
            echo "Unknown option $1"
            print_usage
            exit 1
    esac
done

if [[ ! $KEY ]]; then
    echo "--key must be supplied"
    exit 1
fi

if [[ ! $KEY =~ ^[A-Z_]+$ ]]; then
    echo "Invalid key $KEY"
    exit 1
fi

FILE="$1"

if [ ! -e "$FILE" ]; then
    echo "$FILE does not exist"
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

if [ ! -d "$DIST_DIR" ]; then
    echo "$DIST_DIR does not exist or is not a directory"
    exit 1
fi

FILE_NAME=$(basename "$FILE")

MD5=$(md5 -q "$FILE")

section "Moving $FILE into place" "="

cp "$FILE" "$DIST_DIR"/"$FILE_NAME".new

# Check integrity (buggy sshfs??)

MD5_D=$(md5 -q "$DIST_DIR/$FILE_NAME.new")
if [[ $MD5 != $MD5_D ]]; then
    echo "Error moving $FILE_NAME in place ($MD5 != $MD5_D)!"
    rm "$DIST_DIR"/"$FILE_NAME".new
    exit 1
else
    mv "$DIST_DIR"/"$FILE_NAME".new "$DIST_DIR"/"$FILE_NAME"
fi


section "Registering $KEY=$FILE_NAME with MD5=$MD5" "="

if [[ ! -e "$DIST_DIR"/filenames_mac.set ]]; then
    touch "$DIST_DIR"/filenames_mac.set
fi

egrep -v "^($KEY)=" "$DIST_DIR"/filenames_mac.set > "$DIST_DIR"/filenames_mac.set.new || true


echo "$KEY=$FILE_NAME" >> "$DIST_DIR"/filenames_mac.set.new
echo "$KEY=$FILE_NAME"

mv "$DIST_DIR"/filenames_mac.set.new "$DIST_DIR"/filenames_mac.set

sync

if [[ $UNMOUNT ]]; then
    diskutil unmount "$DIST_DIR"
fi
