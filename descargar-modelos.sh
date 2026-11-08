#!/usr/bin/env bash
# Baja los modelos de los flujos al volumen de red de RunPod.
#
# Se corre UNA vez en un pod temporal que tenga el volumen de red montado
# (en un pod el volumen queda en /workspace; en serverless en /runpod-volume).
#
# Uso:
#   export HF_TOKEN=...           # token de lectura de Hugging Face (no se imprime)
#   ./descargar-modelos.sh todo   # o: wan22 | ltx25 | h3 | imagen
#
# Variables opcionales:
#   VOLUMEN=/workspace            # raíz del volumen (por defecto /workspace o /runpod-volume)
#
# Nombres y rutas verificados el 7/11/2026 (video) y el 8/11/2026 (imagen) con la
# API de Hugging Face (ver el "_meta" de cada archivo en flujos/).
set -euo pipefail

OPCION="${1:-}"
if [[ -z "$OPCION" || ! "$OPCION" =~ ^(wan22|ltx25|h3|imagen|todo)$ ]]; then
  echo "Uso: $0 wan22 | ltx25 | h3 | imagen | todo" >&2
  exit 64
fi

if [[ -z "${VOLUMEN:-}" ]]; then
  if [[ -d /workspace ]]; then VOLUMEN=/workspace
  elif [[ -d /runpod-volume ]]; then VOLUMEN=/runpod-volume
  else
    echo "No encuentro el volumen de red (/workspace ni /runpod-volume). Define VOLUMEN=/ruta." >&2
    exit 1
  fi
fi
MODELOS="$VOLUMEN/models"
TEMPORAL="$VOLUMEN/.descargas-hf"
mkdir -p "$MODELOS" "$TEMPORAL"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "Falta HF_TOKEN. Crea un token de lectura en https://huggingface.co/settings/tokens y corre: export HF_TOKEN=..." >&2
  exit 1
fi
export HF_TOKEN
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"

# CLI de Hugging Face: hoy se llama `hf` (huggingface-cli es el nombre viejo).
# https://huggingface.co/docs/huggingface_hub/guides/cli
if ! command -v hf >/dev/null 2>&1; then
  echo "Instalando el CLI de Hugging Face (hf)…"
  python3 -m pip install -q -U "huggingface_hub" >/dev/null
fi
if ! command -v hf >/dev/null 2>&1; then
  echo "No pude instalar el comando «hf». Prueba: python3 -m pip install -U huggingface_hub" >&2
  exit 1
fi

# Revisa el acceso antes de bajar 20 GB (401/403 = licencia sin aceptar o token malo).
# Sin -L: con acceso, Hugging Face responde 302 hacia su CDN; sin acceso, 401/403.
revisar_acceso() {
  local repo="$1" ruta="$2" codigo
  codigo=$(curl -s -o /dev/null -w "%{http_code}" -I \
    -H "Authorization: Bearer ${HF_TOKEN}" \
    "https://huggingface.co/${repo}/resolve/main/${ruta}" || echo "000")
  case "$codigo" in
    200|302) return 0 ;;
    401|403)
      echo "" >&2
      echo "ERROR ($codigo): no tienes acceso a ${repo}." >&2
      echo "  Acepta la licencia en https://huggingface.co/${repo} (entra con tu cuenta y pulsa el botón para aceptar)" >&2
      echo "  y revisa que HF_TOKEN sea un token de tu cuenta con permiso de lectura." >&2
      exit 2 ;;
    404)
      echo "ERROR: no existe ${repo}/${ruta} (¿cambió el nombre del archivo?)." >&2
      exit 3 ;;
    *)
      echo "Aviso: Hugging Face respondió HTTP ${codigo} al revisar ${repo}; intento bajar igual." >&2 ;;
  esac
}

# bajar <repo> <ruta en el repo> <carpeta en models/>
bajar() {
  local repo="$1" ruta="$2" carpeta="$3"
  local archivo destino
  archivo="$(basename "$ruta")"
  destino="$MODELOS/$carpeta/$archivo"
  mkdir -p "$MODELOS/$carpeta"
  if [[ -s "$destino" ]]; then
    echo "✔ Ya está: models/$carpeta/$archivo"
    return 0
  fi
  revisar_acceso "$repo" "$ruta"
  echo "↓ Bajando $repo / $ruta → models/$carpeta/"
  # --local-dir conserva las subcarpetas del repo; por eso se baja a TEMPORAL y luego se mueve.
  hf download "$repo" "$ruta" --local-dir "$TEMPORAL/$repo" >/dev/null
  mv -f "$TEMPORAL/$repo/$ruta" "$destino"
  echo "✔ Listo: models/$carpeta/$archivo ($(du -h "$destino" | cut -f1))"
}

bajar_wan22() {
  echo "== Wan 2.2 (I2V 14B + LightX2V 4 pasos, TI2V 5B) — Apache 2.0, ~49 GB =="
  bajar Comfy-Org/Wan_2.2_ComfyUI_Repackaged split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors diffusion_models
  bajar Comfy-Org/Wan_2.2_ComfyUI_Repackaged split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors diffusion_models
  bajar Comfy-Org/Wan_2.2_ComfyUI_Repackaged split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors loras
  bajar Comfy-Org/Wan_2.2_ComfyUI_Repackaged split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors loras
  bajar Comfy-Org/Wan_2.2_ComfyUI_Repackaged split_files/diffusion_models/wan2.2_ti2v_5B_fp16.safetensors diffusion_models
  bajar Comfy-Org/Wan_2.2_ComfyUI_Repackaged split_files/vae/wan_2.1_vae.safetensors vae
  bajar Comfy-Org/Wan_2.2_ComfyUI_Repackaged split_files/vae/wan2.2_vae.safetensors vae
  bajar Comfy-Org/Wan_2.1_ComfyUI_repackaged split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors text_encoders
}

bajar_ltx25() {
  echo "== LTX-2.5 destilado int8 — licencia comunitaria de Lightricks (hay que aceptarla), ~40 GB =="
  bajar Lightricks/LTX-2.5 diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors diffusion_models
  bajar Lightricks/LTX-2.5 text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors text_encoders
  bajar Lightricks/LTX-2.5 vae/ltx-2.5-video-vae-bf16.safetensors vae
  bajar Lightricks/LTX-2.5 vae/ltx-2.5-audio-vae-bf16.safetensors vae
  bajar Lightricks/LTX-2.5 latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors latent_upscale_models
}

bajar_h3() {
  echo "== MiniMax H3 (FastH3 OmniRef 8 pasos int8) — licencia de MiniMax con territorios excluidos, ~41 GB =="
  echo "   Recuerda: los videos de H3 NO se pueden mostrar en EE. UU., UE, Reino Unido ni Corea del Sur."
  bajar FastVideo/FastH3-OmniRef-Comfy diffusion_models/fasth3_omniref_8step_pruned_int8_convrot.safetensors diffusion_models
  bajar Comfy-Org/MiniMax-H3 text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors text_encoders
  bajar Comfy-Org/MiniMax-H3 vae/minimax_h3_video_vae_int8_convrot.safetensors vae
  bajar Comfy-Org/MiniMax-H3 vae/minimax_h3_audio_vae_fp32.safetensors vae
}

bajar_imagen() {
  echo "== Imagen: Z-Image Turbo + Qwen-Image-Edit 2511 (fp8mixed) + LoRA Lightning 4/8 pasos — Apache 2.0, ~53 GB =="
  # Comfy-Org renombró estos repos (z_image_turbo → Z-Image-Turbo, Qwen-Image_ComfyUI → Qwen-Image,
  # Qwen-Image-Edit_ComfyUI → Qwen-Image-Edit); se usan los nombres nuevos para no depender de la redirección.
  bajar Comfy-Org/Z-Image-Turbo split_files/diffusion_models/z_image_turbo_bf16.safetensors diffusion_models
  bajar Comfy-Org/Z-Image-Turbo split_files/text_encoders/qwen_3_4b.safetensors text_encoders
  bajar Comfy-Org/Z-Image-Turbo split_files/vae/ae.safetensors vae
  bajar Comfy-Org/Qwen-Image-Edit split_files/diffusion_models/qwen_image_edit_2511_fp8mixed.safetensors diffusion_models
  bajar Comfy-Org/Qwen-Image split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors text_encoders
  bajar Comfy-Org/Qwen-Image split_files/vae/qwen_image_vae.safetensors vae
  # LoRA Lightning (lightx2v) para el flujo qwen-edit-rapido: 4 pasos con cfg 1; la de 8 pasos es el plan B de calidad.
  bajar lightx2v/Qwen-Image-Edit-2511-Lightning Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors loras
  bajar lightx2v/Qwen-Image-Edit-2511-Lightning Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors loras
  # Máquina de catálogos: LoRA Multiángulo (catalogo-angulos) y Real-ESRGAN ×4 (mejorar-2k).
  # El 8/11/2026 ya estaban en el volumen. Real-ESRGAN se baja en .pth: el handler acepta
  # «RealESRGAN_x4plus.pth» aunque el flujo pida «.safetensors» (resolver_extensiones_de_modelo).
  bajar fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA qwen-image-edit-2511-multiple-angles-lora.safetensors loras
  bajar lllyasviel/Annotators RealESRGAN_x4plus.pth upscale_models
}

case "$OPCION" in
  wan22)  bajar_wan22 ;;
  ltx25)  bajar_ltx25 ;;
  h3)     bajar_h3 ;;
  imagen) bajar_imagen ;;
  todo)   bajar_wan22; bajar_ltx25; bajar_h3; bajar_imagen ;;
esac

rm -rf "$TEMPORAL"
echo ""
echo "== Espacio usado en $MODELOS =="
du -sh "$MODELOS"/* 2>/dev/null || true
du -sh "$MODELOS"
df -h "$VOLUMEN" | tail -n 1 || true
