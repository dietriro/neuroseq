PLATFORM="pynn-nest"
BRANCH_CODE="main"
BRANCH_DATA="main"

script_dir=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
path_package="$(dirname "${script_dir}")"

cd ${path_package}/platforms/${PLATFORM} || exit

docker build -t neuroseq:"${PLATFORM}" --build-arg BRANCH_CODE="${BRANCH_CODE}" --build-arg BRANCH_DATA="${BRANCH_DATA}" --network=host --no-cache .
