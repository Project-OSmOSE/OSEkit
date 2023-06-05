

set -ex



pip check
python -c "import bottleneck; bottleneck.test()"
exit 0
