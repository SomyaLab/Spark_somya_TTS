#!/bin/bash

set -e  # Exit on error

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATASETS_DIR="$PROJECT_ROOT/datasets"
TEMP_DIR="$PROJECT_ROOT/temp_downloads"

# Create directories
mkdir -p "$DATASETS_DIR"
mkdir -p "$TEMP_DIR"

# Function to get language code from language name
get_lang_code() {
    local lang_name="$1"
    case "$lang_name" in
        "Gujarati") echo "gu" ;;
        "Bengali") echo "bn" ;;
        "Bhojpuri") echo "bh" ;;
        "Chhattisgarhi") echo "hne" ;;
        "Hindi") echo "hi" ;;
        "Kannada") echo "kn" ;;
        "Magahi") echo "mag" ;;
        "Maithili") echo "mai" ;;
        "Marathi") echo "mr" ;;
        "Telugu") echo "te" ;;
        *) echo "" ;;
    esac
}

# Function to extract language from filename
extract_lang() {
    local filename="$1"
    if [[ "$filename" =~ IISc_(SPICOR|SYSPIN)Project_([A-Za-z]+)_ ]]; then
        echo "${BASH_REMATCH[2]}"
    fi
}

# Function to extract project type from filename
extract_project_type() {
    local filename="$1"
    if [[ "$filename" =~ IISc_(SPICOR|SYSPIN)Project_ ]]; then
        echo "${BASH_REMATCH[1]}"
    fi
}

# Function to download and extract a single file
download_and_extract() {
    local url="$1"
    local filename="$2"
    local temp_file="$TEMP_DIR/$filename"
    
    echo "[$(date +%H:%M:%S)] Downloading $filename..."
    if wget -q --show-progress "$url" -O "$temp_file"; then
        echo "[$(date +%H:%M:%S)] Downloaded $filename"
        
        # Extract language and project type
        local lang_name=$(extract_lang "$filename")
        local project_type=$(extract_project_type "$filename")
        local lang_code=$(get_lang_code "$lang_name")
        
        if [ -z "$lang_code" ]; then
            echo "[$(date +%H:%M:%S)] Warning: Unknown language '$lang_name' in $filename, skipping extraction"
            rm -f "$temp_file"
            return 1
        fi
        
        # Determine data folder name
        local data_folder="IISc_${project_type}_Data"
        
        # Create language directory structure
        local lang_dir="$DATASETS_DIR/$lang_code/$data_folder"
        mkdir -p "$lang_dir"
        
        # Extract to language directory
        # Tar files typically contain a top-level directory, extract it directly
        echo "[$(date +%H:%M:%S)] Extracting $filename to $lang_dir..."
        tar -xzf "$temp_file" -C "$lang_dir"
        
        # Clean up temp file
        rm "$temp_file"
        echo "[$(date +%H:%M:%S)] Completed $filename"
    else
        echo "[$(date +%H:%M:%S)] Error: Failed to download $filename"
        rm -f "$temp_file"
        return 1
    fi
}

# Export functions and variables for parallel execution
export -f download_and_extract extract_lang extract_project_type get_lang_code
export DATASETS_DIR TEMP_DIR

# Array of download URLs and filenames
declare -a downloads=(
    "https://objectstore.e2enetworks.net/iisc-spire-corpora/spicor/gujarati_tts/IISc_SPICORProject_Gujarati_Female_Spk001_HC.tar.gz?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=0U8R7S2207455OVWTNCN%2F20260116%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20260116T204029Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=e14f62c56359c0ec91a32dd1118697613fdbbbdd68eb064e85076db065ab84cf|IISc_SPICORProject_Gujarati_Female_Spk001_HC.tar.gz"
    "https://objectstore.e2enetworks.net/iisc-spire-corpora/spicor/gujarati_tts/IISc_SPICORProject_Gujarati_Female_Spk001_NHC.tar.gz?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=0U8R7S2207455OVWTNCN%2F20260116%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20260116T204029Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=97af5407031fac22e54df85213cd53bb5dd1314fad22788877224165827a0071|IISc_SPICORProject_Gujarati_Female_Spk001_NHC.tar.gz"
    "https://objectstore.e2enetworks.net/iisc-spire-corpora/spicor/gujarati_tts/IISc_SPICORProject_Gujarati_Male_Spk001_HC.tar.gz?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=0U8R7S2207455OVWTNCN%2F20260116%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20260116T204029Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=dd17570dd914c1184d00cd9cfb791110bc9347374980b899b829ddba6c1e6bd0|IISc_SPICORProject_Gujarati_Male_Spk001_HC.tar.gz"
    "https://objectstore.e2enetworks.net/iisc-spire-corpora/spicor/gujarati_tts/IISc_SPICORProject_Gujarati_Male_Spk001_NHC.tar.gz?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=0U8R7S2207455OVWTNCN%2F20260116%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20260116T204029Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=74d9672d6a228badc63bfcb3830018774ae17c6468e0387aac7364fb506c3051|IISc_SPICORProject_Gujarati_Male_Spk001_NHC.tar.gz"
    "https://objectstore.e2enetworks.net/iisc-spire-corpora/syspin/IISc_SYSPINProject_Bengali_Female_Spk001_HC.tar.gz?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=0U8R7S2207455OVWTNCN%2F20260116%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20260116T203938Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=8fc5e9954426614328d1c56050b312b8b95b2dfce2bbaf07425d0c02c9a9f477|IISc_SYSPINProject_Bengali_Female_Spk001_HC.tar.gz"
    "https://objectstore.e2enetworks.net/iisc-spire-corpora/syspin/IISc_SYSPINProject_Bengali_Female_Spk001_NHC.tar.gz?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=0U8R7S2207455OVWTNCN%2F20260116%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20260116T203938Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=a247332c58ecb018bdfca6e24157479a0c1edfefa0661a783d95ea5785f82547|IISc_SYSPINProject_Bengali_Female_Spk001_NHC.tar.gz"
    "https://objectstore.e2enetworks.net/iisc-spire-corpora/syspin/IISc_SYSPINProject_Bengali_Male_Spk001_HC.tar.gz?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=0U8R7S2207455OVWTNCN%2F20260116%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20260116T203938Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=5db77be770d4dd805ef32d25c1f9aec7306e606c897ef5d98f4351365aedfaa8|IISc_SYSPINProject_Bengali_Male_Spk001_HC.tar.gz"
    "https://objectstore.e2enetworks.net/iisc-spire-corpora/syspin/IISc_SYSPINProject_Bengali_Male_Spk001_NHC.tar.gz?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=0U8R7S2207455OVWTNCN%2F20260116%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20260116T203938Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=27daf14719c6b9e3b1dc0e095dbf12fcdd1d8e291f7916fa90455830725190c6|IISc_SYSPINProject_Bengali_Male_Spk001_NHC.tar.gz"
    "https://objectstore.e2enetworks.net/iisc-spire-corpora/syspin/IISc_SYSPINProject_Bhojpuri_Female_Spk001_HC.tar.gz?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=0U8R7S2207455OVWTNCN%2F20260116%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20260116T203938Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=1f9675f3fbc1b6df0d6d103dbde1908db298fd83cd69035ecbc2772af62a4082|IISc_SYSPINProject_Bhojpuri_Female_Spk001_HC.tar.gz"
    "https://objectstore.e2enetworks.net/iisc-spire-corpora/syspin/IISc_SYSPINProject_Bhojpuri_Female_Spk001_NHC.tar.gz?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=0U8R7S2207455OVWTNCN%2F20260116%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20260116T203938Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=c3acd47a6a87afeadf61a41df7ac3f017e6e7c66b74f20fdf03f6251ef9b11f9|IISc_SYSPINProject_Bhojpuri_Female_Spk001_NHC.tar.gz"
    "https://objectstore.e2enetworks.net/iisc-spire-corpora/syspin/IISc_SYSPINProject_Bhojpuri_Male_Spk001_HC.tar.gz?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=0U8R7S2207455OVWTNCN%2F20260116%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20260116T203938Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=c2c572d3826ef4e4e749c2ddfc93b90ae1a5f5406b48ad263046a03f0a246b63|IISc_SYSPINProject_Bhojpuri_Male_Spk001_HC.tar.gz"
    "https://objectstore.e2enetworks.net/iisc-spire-corpora/syspin/IISc_SYSPINProject_Bhojpuri_Male_Spk001_NHC.tar.gz?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=0U8R7S2207455OVWTNCN%2F20260116%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20260116T203938Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=8e1b6854b1f283376848fae4f64b58496b862632b4e4ea2ce5fb79487019a697|IISc_SYSPINProject_Bhojpuri_Male_Spk001_NHC.tar.gz"
    "https://objectstore.e2enetworks.net/iisc-spire-corpora/syspin/IISc_SYSPINProject_Chhattisgarhi_Female_Spk001_HC.tar.gz?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=0U8R7S2207455OVWTNCN%2F20260116%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20260116T203938Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=643f16364d5a11a088ec1af1e0d2c6d8b629144b8148c110e2f440cc5e256913|IISc_SYSPINProject_Chhattisgarhi_Female_Spk001_HC.tar.gz"
    "https://objectstore.e2enetworks.net/iisc-spire-corpora/syspin/IISc_SYSPINProject_Chhattisgarhi_Female_Spk001_NHC.tar.gz?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=0U8R7S2207455OVWTNCN%2F20260116%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20260116T203938Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=4eb8d03b39084a598ba638ed3339330553c02a61fff4ccdb2788f2f487d6a61e|IISc_SYSPINProject_Chhattisgarhi_Female_Spk001_NHC.tar.gz"
    "https://objectstore.e2enetworks.net/iisc-spire-corpora/syspin/IISc_SYSPINProject_Chhattisgarhi_Male_Spk001_HC.tar.gz?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=0U8R7S2207455OVWTNCN%2F20260116%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20260116T203938Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=da2f311a775b8f19d617dc28232872b7ccf921aac7dea49e36b7917556f2e2b0|IISc_SYSPINProject_Chhattisgarhi_Male_Spk001_HC.tar.gz"
    "https://objectstore.e2enetworks.net/iisc-spire-corpora/syspin/IISc_SYSPINProject_Chhattisgarhi_Male_Spk001_NHC.tar.gz?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=0U8R7S2207455OVWTNCN%2F20260116%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20260116T203938Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=e5105561ad583f95c5a35f6fb5ea36927e45a352bdcedfb089aae706bc5159de|IISc_SYSPINProject_Chhattisgarhi_Male_Spk001_NHC.tar.gz"
    "https://objectstore.e2enetworks.net/iisc-spire-corpora/syspin/IISc_SYSPINProject_Hindi_Female_Spk001_HC.tar.gz?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=0U8R7S2207455OVWTNCN%2F20260116%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20260116T203938Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=e31eb3827d265c56fa6a3d608de52597ea0cc8aad5eae875ca1dc7049edba354|IISc_SYSPINProject_Hindi_Female_Spk001_HC.tar.gz"
    "https://objectstore.e2enetworks.net/iisc-spire-corpora/syspin/IISc_SYSPINProject_Hindi_Female_Spk001_NHC.tar.gz?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=0U8R7S2207455OVWTNCN%2F20260116%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20260116T203938Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=feedd08b7e6cc78a586ec97a29cd38086a54ddcd05300d998114809b17c338f2|IISc_SYSPINProject_Hindi_Female_Spk001_NHC.tar.gz"
    "https://objectstore.e2enetworks.net/iisc-spire-corpora/syspin/IISc_SYSPINProject_Hindi_Male_Spk001_HC.tar.gz?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=0U8R7S2207455OVWTNCN%2F20260116%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20260116T203938Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=c2d09c2726f8de7f297cf484c4f4c2e00f92973256ecbc70a0fe0f4cc5aad629|IISc_SYSPINProject_Hindi_Male_Spk001_HC.tar.gz"
    "https://objectstore.e2enetworks.net/iisc-spire-corpora/syspin/IISc_SYSPINProject_Hindi_Male_Spk001_NHC.tar.gz?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=0U8R7S2207455OVWTNCN%2F20260116%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20260116T203938Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=6ded7f47179af8acfe58577bdb91d3971d06a1144f5a3bc46dfec7357b37ec45|IISc_SYSPINProject_Hindi_Male_Spk001_NHC.tar.gz"
    "https://objectstore.e2enetworks.net/iisc-spire-corpora/syspin/IISc_SYSPINProject_Kannada_Female_Spk001_HC.tar.gz?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=0U8R7S2207455OVWTNCN%2F20260116%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20260116T203939Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=6bc9ff2ee1a48ecde9fe2118b799d2d9c372b13f6b43ee89a4b742c8ac9afd66|IISc_SYSPINProject_Kannada_Female_Spk001_HC.tar.gz"
    "https://objectstore.e2enetworks.net/iisc-spire-corpora/syspin/IISc_SYSPINProject_Kannada_Female_Spk001_NHC.tar.gz?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=0U8R7S2207455OVWTNCN%2F20260116%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20260116T203939Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=6db01d6c18fec12c706328650e8460afe6bfd1801e7b2d9d6df88e8bff36ec78|IISc_SYSPINProject_Kannada_Female_Spk001_NHC.tar.gz"
    "https://objectstore.e2enetworks.net/iisc-spire-corpora/syspin/IISc_SYSPINProject_Kannada_Male_Spk001_HC.tar.gz?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=0U8R7S2207455OVWTNCN%2F20260116%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20260116T203939Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=0574a4cd39ce2aa6ad61ce088f50779428b97b30ab1a6ba51cb24379cd18ae41|IISc_SYSPINProject_Kannada_Male_Spk001_HC.tar.gz"
    "https://objectstore.e2enetworks.net/iisc-spire-corpora/syspin/IISc_SYSPINProject_Kannada_Male_Spk001_NHC.tar.gz?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=0U8R7S2207455OVWTNCN%2F20260116%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20260116T203939Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=47502789d5c9fbf76e71e7d0508f10084678b77356a8bece80e7300c7f62e2f4|IISc_SYSPINProject_Kannada_Male_Spk001_NHC.tar.gz"
    "https://objectstore.e2enetworks.net/iisc-spire-corpora/syspin/IISc_SYSPINProject_Magahi_Female_Spk001_HC.tar.gz?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=0U8R7S2207455OVWTNCN%2F20260116%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20260116T203939Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=14f80068613f43791225047ece33445dad9202a50dbd755acae6f16f1666bf31|IISc_SYSPINProject_Magahi_Female_Spk001_HC.tar.gz"
    "https://objectstore.e2enetworks.net/iisc-spire-corpora/syspin/IISc_SYSPINProject_Magahi_Female_Spk001_NHC.tar.gz?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=0U8R7S2207455OVWTNCN%2F20260116%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20260116T203939Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=76485359c278013097d5158da95c64d7c8fcfb20b1ed871e1cabe50b8dea63a5|IISc_SYSPINProject_Magahi_Female_Spk001_NHC.tar.gz"
    "https://objectstore.e2enetworks.net/iisc-spire-corpora/syspin/IISc_SYSPINProject_Magahi_Male_Spk001_HC.tar.gz?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=0U8R7S2207455OVWTNCN%2F20260116%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20260116T203939Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=dac1701b8ca7f9af9dc136bef8fead58c5a2f46ecdb04258870c82c4a057654b|IISc_SYSPINProject_Magahi_Male_Spk001_HC.tar.gz"
    "https://objectstore.e2enetworks.net/iisc-spire-corpora/syspin/IISc_SYSPINProject_Magahi_Male_Spk001_NHC.tar.gz?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=0U8R7S2207455OVWTNCN%2F20260116%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20260116T203939Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=70f557b4d388739da786d8beca6e2c15a4ca3ea408978d82bc58a84aa80f15ef|IISc_SYSPINProject_Magahi_Male_Spk001_NHC.tar.gz"
    "https://objectstore.e2enetworks.net/iisc-spire-corpora/syspin/IISc_SYSPINProject_Maithili_Female_Spk001_HC.tar.gz?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=0U8R7S2207455OVWTNCN%2F20260116%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20260116T203939Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=d101de20fbf7b7d5642037948ef69367547f9e39729b3740e91af3e903056c78|IISc_SYSPINProject_Maithili_Female_Spk001_HC.tar.gz"
    "https://objectstore.e2enetworks.net/iisc-spire-corpora/syspin/IISc_SYSPINProject_Maithili_Female_Spk001_NHC.tar.gz?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=0U8R7S2207455OVWTNCN%2F20260116%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20260116T203939Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=3abc4fabf93055cc28565ccdf26c05adcbfbed7c7db98ba44215a317fbe416ed|IISc_SYSPINProject_Maithili_Female_Spk001_NHC.tar.gz"
    "https://objectstore.e2enetworks.net/iisc-spire-corpora/syspin/IISc_SYSPINProject_Maithili_Male_Spk001_HC.tar.gz?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=0U8R7S2207455OVWTNCN%2F20260116%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20260116T203939Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=9beccf41d67c2aad4e915e304080a66b8dae2c61e585c3f6224c0cc3309846e9|IISc_SYSPINProject_Maithili_Male_Spk001_HC.tar.gz"
    "https://objectstore.e2enetworks.net/iisc-spire-corpora/syspin/IISc_SYSPINProject_Maithili_Male_Spk001_NHC.tar.gz?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=0U8R7S2207455OVWTNCN%2F20260116%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20260116T203939Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=cb6f617c073554975fa7826118c88900711278754e9fc975897f488914c7b519|IISc_SYSPINProject_Maithili_Male_Spk001_NHC.tar.gz"
    "https://objectstore.e2enetworks.net/iisc-spire-corpora/syspin/IISc_SYSPINProject_Marathi_Female_Spk001_HC.tar.gz?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=0U8R7S2207455OVWTNCN%2F20260116%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20260116T203939Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=c44cd3aec18a8d154a8b92f3671d4aff5d4399a6f8d7b5f288496a32cf161b40|IISc_SYSPINProject_Marathi_Female_Spk001_HC.tar.gz"
    "https://objectstore.e2enetworks.net/iisc-spire-corpora/syspin/IISc_SYSPINProject_Marathi_Female_Spk001_NHC.tar.gz?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=0U8R7S2207455OVWTNCN%2F20260116%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20260116T203939Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=8645fb21eb96aaba07f885ab268b11d73195af7ffc2ba82644e71fdc2ee2339e|IISc_SYSPINProject_Marathi_Female_Spk001_NHC.tar.gz"
    "https://objectstore.e2enetworks.net/iisc-spire-corpora/syspin/IISc_SYSPINProject_Marathi_Male_Spk001_HC.tar.gz?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=0U8R7S2207455OVWTNCN%2F20260116%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20260116T203939Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=a25ce4cb7186c09f7f4d0317647c0c7b71317fc7b34fb362229d180989e8b4ae|IISc_SYSPINProject_Marathi_Male_Spk001_HC.tar.gz"
    "https://objectstore.e2enetworks.net/iisc-spire-corpora/syspin/IISc_SYSPINProject_Marathi_Male_Spk001_NHC.tar.gz?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=0U8R7S2207455OVWTNCN%2F20260116%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20260116T203939Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=57987b81c9d88f19f2af06eed7e2f13612f8f02837f8ca07be0b47bfa8c747f7|IISc_SYSPINProject_Marathi_Male_Spk001_NHC.tar.gz"
    "https://objectstore.e2enetworks.net/iisc-spire-corpora/syspin/IISc_SYSPINProject_Telugu_Female_Spk001_HC.tar.gz?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=0U8R7S2207455OVWTNCN%2F20260116%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20260116T203939Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=c75251d6fb4a62e971c3f258ce24bdbc27bdecbd874ee0a1b8f9578ec07eed2a|IISc_SYSPINProject_Telugu_Female_Spk001_HC.tar.gz"
    "https://objectstore.e2enetworks.net/iisc-spire-corpora/syspin/IISc_SYSPINProject_Telugu_Female_Spk001_NHC.tar.gz?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=0U8R7S2207455OVWTNCN%2F20260116%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20260116T203939Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=81439b6693b19d75850687f00aea00bc8c6ba5749bc8739b9e66067530722625|IISc_SYSPINProject_Telugu_Female_Spk001_NHC.tar.gz"
    "https://objectstore.e2enetworks.net/iisc-spire-corpora/syspin/IISc_SYSPINProject_Telugu_Male_Spk001_HC.tar.gz?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=0U8R7S2207455OVWTNCN%2F20260116%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20260116T203939Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=530a7ae63a1e52307af3437e08680e80833f0d7e70a9ff8912feefca7e6b3629|IISc_SYSPINProject_Telugu_Male_Spk001_HC.tar.gz"
    "https://objectstore.e2enetworks.net/iisc-spire-corpora/syspin/IISc_SYSPINProject_Telugu_Male_Spk001_NHC.tar.gz?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=0U8R7S2207455OVWTNCN%2F20260116%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20260116T203939Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=fcba996333db1489d0dd4b9501036806447755104d56c2d127069c3eae9c7df3|IISc_SYSPINProject_Telugu_Male_Spk001_NHC.tar.gz"
)

# Function to process download entry
process_download() {
    local entry="$1"
    local url="${entry%%|*}"
    local filename="${entry#*|}"
    download_and_extract "$url" "$filename"
}

export -f process_download

# Download in parallel (max 8 concurrent downloads)
echo "Starting parallel downloads (8 concurrent)..."
printf '%s\n' "${downloads[@]}" | xargs -n 1 -P 8 -I {} bash -c 'process_download "$@"' _ {}

# Clean up temp directory
if [ -d "$TEMP_DIR" ]; then
    rm -rf "$TEMP_DIR"
fi

echo ""
echo "========================================="
echo "All downloads completed!"
echo "Data organized in: $DATASETS_DIR"
echo "========================================="
