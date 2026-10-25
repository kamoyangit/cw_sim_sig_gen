#!/bin/bash
# CWデータセット自動生成スクリプト

set -euo pipefail

# スクリプトの場所を基準にする。相対パスのままだと別ディレクトリから実行したときに
# 黙って失敗する。
cd "$(dirname "$0")"

GEN_CW="./build/gen_cw"
TEXT_FILE="corpus.txt"
OUT_DIR="./dataset_cw_train"

if [ ! -f "$GEN_CW" ]; then
	echo "Error: $GEN_CW が見つかりません。make -C tools/cwsim を実行してください。"
	exit 1
fi

if [ ! -f "$TEXT_FILE" ]; then
	echo "Error: $TEXT_FILE が見つかりません。"
	exit 1
fi

mkdir -p "$OUT_DIR"

# データセットの自己記述。特に S/N の基準帯域は、ファイル名だけ見ると必ず誤読される。
write_readme() {
	{
		echo "# CW dataset"
		echo
		echo "Generated $(date -u +%Y-%m-%dT%H:%M:%SZ) by \`generate_dataset.sh\` from \`${TEXT_FILE}\`."
		echo
		echo "## S:N: the number in the file name is the 3 kHz figure"
		echo
		echo "\`--snr\` is the ratio of key-down signal power to total noise power **inside a"
		echo "3 kHz audio band**, measured over the whole file."
		echo
		echo "A CW receiver does not listen in 3 kHz. The same file measures higher in a"
		echo "narrow filter, because less noise is included:"
		echo
		echo "| file name | 3 kHz (audio band) | 250 Hz filter | 500 Hz filter |"
		echo "|---|---|---|---|"
		for s in "${SNRS[@]}"; do
			awk -v s="$s" 'BEGIN{
				printf "| snr%ddB | %d dB | %+.1f dB | %+.1f dB |\n", s, s, s+10.79, s+7.78
			}'
		done
		echo
		echo "Quoting a decoder's performance without the bandwidth is meaningless."
		echo
		echo "## Naming"
		echo
		echo "\`cw_<channel>_wpm<speed>_snr<snr>dB_s<seed>.wav\`"
		echo
		echo "The seed is derived from the parameter triple, so adding or removing"
		echo "combinations does not change the seed of any other file."
		echo
		echo "## Channels"
		echo
		echo "| channel | Doppler spread (2 sigma) | meaning |"
		echo "|---|---|---|"
		echo "| WGN | 0 | AWGN only, no fading and no multipath |"
		echo "| MPG | 0.1 Hz | CCIR multipath good |"
		echo "| MPM | 0.5 Hz | CCIR multipath moderate |"
		echo "| MPP | 1.0 Hz | CCIR multipath poor |"
		echo "| MPD | 2.0 Hz | multipath disturbed |"
		echo
		echo "All multipath channels use \`--delay-preset cw-mid\` (0 / 15 / 35 ms, three"
		echo "paths). Those delays are deliberately longer than the CCIR data-mode values: at"
		echo "20 wpm a dit is 60 ms, so a 4 ms echo is only 7% of a symbol and inaudible, while"
		echo "real HF CW echo is a 10-100 ms effect."
		echo
		echo "## Labels (\`.json\`)"
		echo
		echo "| field | meaning |"
		echo "|---|---|"
		echo "| \`text\` | the exact message |"
		echo "| \`channel\` | mode, paths, delays, Doppler, seed, realised mean power gain |"
		echo "| \`snr\` | requested and measured S:N, plus the equivalent in 250 and 500 Hz |"
		echo "| \`characters[]\` | per character: sample range and \`min_gain_db\` |"
		echo "| \`symbols[]\` | per element (dit/dah/gaps): sample range |"
		echo
		echo "\`min_gain_db\` is the weakest 10 ms key-down RMS inside that character, relative"
		echo "to the long-term average, measured **before** noise is added. It is a property of"
		echo "the channel, not of any decoder."
		echo
		echo "WAV is 32-bit float. Peaks exceed 1.0 because fading adds gain on top of the"
		echo "mean-normalised level; do not clip."
		echo
		echo "## Known limitations"
		echo
		echo "- No receiver AGC. A real AGC lifts the noise during deep fades and this does not."
		echo "- No transmitter defects beyond risetime, timing jitter and tone drift."
		echo "- No QRM (a second CW signal) and no QRN (impulse noise, mains hum)."
		echo "- Keying uses the ideal dah:dit ratio of 3.00 unless \`--dah-ratio\` is given; a"
		echo "  measured 40 m signal gave 2.72."
		echo "- The generator is a Watterson/CCIR channel verified against the model and against"
		echo "  one real 40 m capture. It is not claimed to be statistically identical to the"
		echo "  bands."
	} > "${OUT_DIR}/README.md"
}


# 変数パラメータの定義（グリッドスイープ用）
CHANNELS=("WGN" "MPG" "MPM" "MPP" "MPD") # 電離層チャネル（白色雑音〜重度フェンジング）
WPMS=(15 20 25)                    # モールス送信速度 (WPM)
SNRS=(-3 0 3 6 10)                 # S/N比 (dB, 3 kHz 音声帯)

# WGN は「フェージングなし」なので遅延プリセットを渡さない。渡すと静的マルチパス
# （くし形フィルタ）になり、純粋な白色雑音でなくなる。空配列を set -u の下で展開すると
# bash 3.2（macOS 既定）ではエラーになるため、呼び出しを分けている。
run_one() {
	local ch="$1" wpm="$2" snr="$3" seed="$4" wav="$5" json="$6"
	local -a common=(
		--text-file "$TEXT_FILE"
		--repeat 2
		--channel "$ch"
		--wpm "$wpm"
		--snr "$snr"
		--seed "$seed"
		--out "$wav"
		--labels "$json"
	)
	if [ "$ch" = "WGN" ]; then
		"$GEN_CW" "${common[@]}"
	else
		"$GEN_CW" "${common[@]}" --delay-preset cw-mid
	fi
}

count=0
failed=0

write_readme
echo "=== データセット生成開始 ==="

for ch in "${CHANNELS[@]}"; do
	for wpm in "${WPMS[@]}"; do
		for snr in "${SNRS[@]}"; do

			# シードはパラメータから決める。連番にするとチャネルを 1 つ足しただけで
			# 以降のシードがずれ、既に生成したファイルと衝突する。
			seed=$((1000 + $(printf '%s' "${ch}_${wpm}_${snr}" | cksum | cut -d' ' -f1) % 900000))

			base_name="cw_${ch}_wpm${wpm}_snr${snr}dB_s${seed}"
			wav_out="${OUT_DIR}/${base_name}.wav"
			json_out="${OUT_DIR}/${base_name}.json"

			echo "[${count}] Generating: ${base_name}"

			if ! run_one "$ch" "$wpm" "$snr" "$seed" "$wav_out" "$json_out"; then
				echo "  FAILED: ${base_name}" >&2
				failed=$((failed + 1))
			fi

			count=$((count + 1))

		done
	done
done

echo "=== 生成完了: 合計 ${count} 組 (.wav / .json) ==="
if [ "$failed" -gt 0 ]; then
	echo "警告: ${failed} 件が失敗しました。" >&2
	exit 1
fi
