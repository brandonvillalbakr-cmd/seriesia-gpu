# Trabajador GPU de Series IA (RunPod Serverless + ComfyUI)

Esta carpeta arma la imagen Docker que genera video (e imágenes fijas) con modelos abiertos en una GPU alquilada de RunPod. La app la llama con el proveedor `servidor-propio` (ver `docs/10-SERVIDOR-GPU-PROPIO.md`, secciones 5 y 8).

| Archivo | Para qué sirve |
|---|---|
| `Dockerfile` | Imagen para RunPod Serverless: `runpod/worker-comfyui:5.10.0-base` + ComfyUI v0.44.2 + nuestro handler |
| `handler.py` | Recibe el pedido de la app, arma el flujo, lo corre en ComfyUI y devuelve el MP4 (o el PNG en los flujos de imagen) |
| `flujos/*.json` | Los 6 flujos de ComfyUI (4 de video y 2 de imagen) (formato API) con marcadores `{{…}}`; cada uno trae `_meta` con fuentes, modelos, VRAM y TODO |
| `extra_model_paths.yaml` | Le dice a ComfyUI que los modelos están en el volumen de red (`/runpod-volume/models/…`) |
| `descargar-modelos.sh` | Baja los modelos de Hugging Face al volumen de red (se corre una vez en un pod temporal) |
| `pruebas/` | Pruebas sin GPU (`test_handler.py`) y pedidos de ejemplo para `curl` (`entrada-ejemplo.json` de video, `entrada-ejemplo-imagen.json` de imagen) |

> Los rótulos de botones van «entre comillas» tal como aparecen en la documentación oficial que se cita al lado. Donde no pude confirmarlo, lo digo.

---

## Los flujos

### Video (4)

| `flujo` | Modelo | Necesita | Audio | fps | Frames | Tamaño 9:16 real | Modelos en el volumen |
|---|---|---|---|---|---|---|---|
| `h3-ref` | MiniMax H3 · FastH3 OmniRef 8 pasos (int8) | `referencias` (1–9) **o** `imagen_inicial` | Nativo; `audio_referencia` opcional (voz del TTS) | 24 fijo | 17k+5 (124 ≈ 5,2 s) | 768×1344 | ~41 GB |
| `ltx25-i2v` | LTX-2.5 22B destilado (int8), 2 etapas | `imagen_inicial` | Nativo; `audio_referencia` opcional | 24 (o el que mandes) | 8n+1 (121 = 5 s) | **704**×1280 (múltiplo de 64) | ~40 GB |
| `wan22-i2v` | Wan 2.2 A14B + LoRA LightX2V 4 pasos (fp8) | `imagen_inicial` | No | 16 | 4n+1 (81 = 5 s) | 720×1280 | ~38 GB |
| `wan22-ti2v-5b` | Wan 2.2 TI2V 5B (fp16) | Nada (`imagen_inicial` opcional) | No | **24** (así viene la plantilla oficial del 5B) | 4n+1 (121 = 5 s) | **704**×1280 (múltiplo de 32) | ~18 GB (comparte el umt5) |

Total para los 4 de video: **~130 GB**.

### Imagen (2)

| `flujo` | Modelo | Necesita | Pasos | Tamaño 9:16 por defecto | Modelos en el volumen |
|---|---|---|---|---|---|
| `zimage-t2i` | Z-Image Turbo 6B (bf16) | Solo `prompt` | 8, cfg 1 (`res_multistep`) | 768×1344 (múltiplo de 16) | ~20,7 GB |
| `qwen-edit-ref` | Qwen-Image-Edit 2511 (fp8mixed) | `referencias` (1–3) **o** `imagen_inicial` | 40, cfg 4 (`euler`) | 768×1344 (múltiplo de 16) | ~30,2 GB |
| `qwen-edit-rapido` | Qwen-Image-Edit 2511 (fp8mixed) + LoRA Lightning 4 pasos (lightx2v) | igual que `qwen-edit-ref` | 4, cfg 1 (`euler`) | 768×1344 (múltiplo de 16) | +0,85 GB (la LoRA, en `loras/`) |

Total para los 2 de imagen: **~51 GB**. **Los 6 flujos suman ~181 GB**: caben en el volumen de 200 GB, pero queda poco margen (agrándalo si se suman más modelos; se puede agrandar, no achicar).

- `zimage-t2i`: plantilla oficial `image_z_image_turbo` tal cual (UNETLoader → ModelSamplingAuraFlow shift 3 → KSampler; CLIPLoader tipo `lumina2` con `qwen_3_4b`; VAE `ae`). **Ignora `prompt_negativo`**: la plantilla usa cfg 1 con el negativo en cero (ConditioningZeroOut), así que un negativo no cambiaría nada.
- `qwen-edit-ref`: plantilla oficial `image_qwen_image_edit_2511` con sus valores por defecto (sin la LoRA Lightning). Las referencias entran en `image1..image3` de los dos `TextEncodeQwenImageEditPlus` (positivo y negativo). **En el prompt, nómbralas «Picture 1», «Picture 2», «Picture 3»** en el orden de `referencias` (así las rotula el nodo). Única diferencia con la plantilla: el latente es `EmptySD3LatentImage` con el `ancho`/`alto` del pedido (la plantilla toma el tamaño de la imagen 1), para que salga 9:16 aunque las hojas tengan otra forma.

- `qwen-edit-rapido`: el mismo grafo de `qwen-edit-ref` con el interruptor «Enable 4steps LoRA?» de la plantilla encendido: `LoraLoaderModelOnly` (fuerza 1) después de `CFGNorm`, KSampler 4 pasos y cfg 1. Pensado para el probador de ropa (clienta en Picture 1, prenda en Picture 2). Con cfg 1 el `prompt_negativo` casi no influye. Plan B de calidad: la LoRA de 8 pasos ya está en el volumen.

Todos usan **solo nodos nativos de ComfyUI** (las plantillas oficiales de Comfy-Org ya no necesitan nodos personalizados para H3, LTX-2.5, Wan 2.2, Z-Image ni Qwen-Image-Edit). La imagen igual instala VideoHelperSuite y KJNodes con commit fijo, por si luego se usan.

---

## Contrato con la app (fijo)

`POST https://api.runpod.ai/v2/{ENDPOINT_ID}/run` con `{"input": {...}}` y luego `GET /v2/{ENDPOINT_ID}/status/{id}`.

Entrada: `flujo`, `prompt`, `prompt_negativo?`, `duracion_seg`, `aspecto`, `ancho`, `alto`, `fps?`, `semilla?`, `imagen_inicial?`, `referencias?`, `audio_referencia?`, `generar_audio`, `salida?` (`url_subida`, `url_publica?`). Los archivos van como `{"nombre","mime","base64"}` o `{"nombre","mime","url"}`.

Salida: `{"video_base64"}` o `{"video_url"}` + `mime`, `flujo`, `ancho`, `alto`, `duracion_seg`, `fps`, `segundos_gpu`. Además devuelve `frames` y `semilla` (campos extra, se pueden ignorar). En error: `{"error": "mensaje en español"}` → RunPod marca el trabajo `FAILED`.

**Flujos de imagen** (`zimage-t2i`, `qwen-edit-ref`): misma entrada, pero **sin** `duracion_seg`, `fps`, `audio_referencia` ni `generar_audio` (si llegan, se ignoran). Usan `flujo`, `prompt`, `prompt_negativo?` (solo Qwen), `ancho?`/`alto?` (por defecto 768×1344), `semilla?`, `referencias?` (1–3, solo Qwen; si no hay, la `imagen_inicial` hace de referencia 1) y `salida?`. Salida: `{"imagen_base64"}` o `{"imagen_url"}` + `mime` (`image/png`), `flujo`, `ancho`, `alto`, `semilla`, `segundos_gpu`. Con `salida.url_subida` el PUT va con `Content-Type: image/png`.

Cosas que el adaptador TypeScript debe saber:
- **`ancho`/`alto`/`duracion_seg`/`fps` de la respuesta son los reales** (p. ej. LTX devuelve 704×1280 y H3 6 s → 158 frames = 6,58 s). Usa esos para el montaje.
- **H3 nombra las imágenes `<Picture 1>`, `<Picture 2>`…** en el orden de `referencias`. Si tu prompt solo usa `<Subject 1>…` y no trae `subject_definitions:`, el handler antepone un bloque que une `<Subject i>` con `<Picture i>`. Si mandas el prompt ya en el formato oficial de MiniMax, no se toca.
- `prompt_negativo` lo usan LTX y Wan; H3 (FastH3) no usa negativo. Si no lo mandas, va el de la plantilla oficial.
- Límites de RunPod: el cuerpo de `/run` admite **10 MB** y el de `/runsync` **20 MB**; los resultados de `/run` se guardan **30 minutos** ([referencia de operaciones](https://docs.runpod.io/serverless/endpoints/operation-reference)). Para 9 hojas de personaje o un video largo, manda **URLs** en vez de base64 y usa `salida.url_subida`.

---

## Pasos para el dueño (sin parte técnica)

### (a) Cuenta de RunPod y API Key
1. Entra a https://www.runpod.io y regístrate. Consola: https://console.runpod.io
2. Pon la tarjeta y recarga **US$25** (es lo de la prueba de 1 día, sección 8 del documento 10). *No pude confirmar en la documentación el rótulo exacto del botón de recarga; está en la parte de facturación de la consola.*
3. Crea la llave ([documentación](https://docs.runpod.io/get-started/credentials)):
   - Ve a la página **Credentials** (https://console.runpod.io/user/credentials).
   - Abre la pestaña «**API Keys**» y pulsa «**Create API Key**».
   - Ponle un nombre (por ejemplo `series-ia`) y en permisos elige «**All**» (o «**Restricted**» con «**Read/Write**» para Serverless).
   - Pulsa «**Create**» y luego haz clic en la llave para copiarla. **RunPod no la vuelve a mostrar**: pégala de una vez en la app (paso g).

### (b) Hugging Face: licencias y token
1. Crea la cuenta en https://huggingface.co/join
2. **Acepta la licencia de LTX-2.5:** entra a https://huggingface.co/Lightricks/LTX-2.5 con tu cuenta y acepta el formulario («**Agree**», según la [documentación de modelos con acceso restringido](https://huggingface.co/docs/hub/models-gated); el texto exacto del botón lo pone cada autor). Lo comprobé: sin aceptar, la descarga da error 401.
3. **MiniMax H3:** los archivos que usamos (`Comfy-Org/MiniMax-H3` y `FastVideo/FastH3-OmniRef-Comfy`) **hoy no piden aceptar nada** en Hugging Face, pero **la licencia de MiniMax igual aplica** (ver la advertencia del final). Si quieres dejar constancia, entra también a https://huggingface.co/MiniMaxAI/MiniMax-H3 y acéptala.
4. **Token de lectura** ([documentación](https://huggingface.co/docs/hub/security-tokens)): ve a https://huggingface.co/settings/tokens → botón «**New token**» → elige el rol «**read**» y un nombre → créalo y cópialo.

### (c) Volumen de red de 200 GB
[Documentación](https://docs.runpod.io/storage/network-volumes):
1. Ve a la página **Storage** (https://www.console.runpod.io/user/storage).
2. Pulsa «**New Network Volume**».
3. Elige un **data center** que tenga las GPU de abajo (H100 o RTX PRO 6000 y alguna barata para el pod de descarga), ponle de nombre `series-ia-modelos` y **200** GB.
4. Tipo de almacenamiento: «**Standard**» (US$0,07/GB/mes → ~US$14/mes por 200 GB).
5. Pulsa «**Create Network Volume**».

> El volumen amarra el endpoint a ese data center. Se puede agrandar después, pero no achicar.

### (d) Pod temporal para bajar los modelos
[Documentación del volumen en pods](https://docs.runpod.io/storage/network-volumes) y [terminal web](https://docs.runpod.io/pods/connect-to-a-pod):
1. Ve a **Pods** (https://www.console.runpod.io/pods) y pulsa «**Deploy**».
2. Elige «**Network Volume**» → `series-ia-modelos`.
3. Elige la GPU **más barata** que aparezca en ese data center (solo vamos a descargar; no se usa la GPU).
4. Plantilla: cualquiera oficial de RunPod con Python (por ejemplo la de PyTorch). *No confirmé el nombre exacto de la plantilla en la documentación.*
5. Pulsa «**Deploy On-Demand**».
6. Cuando arranque: despliega el pod, pulsa «**Connect**», luego «**Start**» (si la terminal está detenida) y «**Open Web Terminal**».
7. Sube `descargar-modelos.sh` al pod. Lo más fácil: en «**Connect**» abre «**Jupyter Lab**» (bajo «**HTTP Services**», si la plantilla lo trae) y arrastra el archivo a la carpeta `/workspace`. O pégalo con `cat > /workspace/descargar-modelos.sh` en la terminal.
8. En la terminal web, escribe esto (el token **no** se ve al pegarlo):
   ```bash
   cd /workspace
   read -rs HF_TOKEN && export HF_TOKEN     # pega el token de Hugging Face y pulsa Enter
   chmod +x descargar-modelos.sh
   ./descargar-modelos.sh todo              # o solo: wan22 | ltx25 | h3 | imagen
   ```
   Tarda según la red del data center (son ~181 GB con `todo`; ~51 GB solo `imagen`). Si ya bajó algo, lo salta. Si falta aceptar una licencia, se detiene con «Acepta la licencia en https://huggingface.co/…». Al final muestra el espacio usado.
9. **Apaga y borra el pod** cuando termine (el volumen y los modelos se quedan). En el pod el volumen se ve en `/workspace`; en serverless, en `/runpod-volume`.

### (e) Construir y subir la imagen Docker
**Opción recomendada: Docker Hub** ([documentación](https://docs.runpod.io/serverless/workers/deploy)). En una PC con Docker (la construcción baja ~10 GB):
```bash
cd E:/series-ia/gpu
docker build --platform linux/amd64 -t TU_USUARIO_DOCKER/seriesia-gpu:0.1.0 .
docker login
docker push TU_USUARIO_DOCKER/seriesia-gpu:0.1.0
```
- Usa siempre una versión (`0.1.0`, `0.1.1`…), **nunca `:latest`** (RunPod guarda la imagen en caché y no verías los cambios).
- **Construye con la carpeta `gpu/` como contexto** (como arriba). El `.dockerignore` de la raíz del repo excluye `gpu`, así que construir desde la raíz falla.

**Opción GitHub de RunPod** ([documentación](https://docs.runpod.io/serverless/workers/github-integration)): en Settings → tarjeta «**GitHub**» → «**Connect**»; luego «**New Endpoint**» → «**Import Git Repository**», rama y «**Dockerfile Path**» = `gpu/Dockerfile`. Para actualizar hay que crear un *release* en GitHub. Límites: `docker build` máx. **30 min** e imagen máx. **80 GB**. *TODO: la documentación no dice si el contexto de construcción es la raíz del repo o la carpeta del Dockerfile; si es la raíz, el `.dockerignore` de la raíz (que excluye `gpu`) lo rompe. Por eso recomiendo Docker Hub.*

### (f) Crear el endpoint Serverless
[Documentación](https://docs.runpod.io/serverless/workers/deploy) y [ajustes](https://docs.runpod.io/serverless/endpoints/endpoint-configurations):
1. Ve a **Serverless** (https://www.console.runpod.io/serverless) y pulsa «**New Endpoint**».
2. Elige «**Import from Docker Registry**» (en otra página de la documentación la misma opción aparece como la tarjeta «**Docker**»), pega en «**Container Image**» `docker.io/TU_USUARIO_DOCKER/seriesia-gpu:0.1.0` y pulsa «**Next**».
3. «**Endpoint Name**»: `series-ia-video`. «**Endpoint Type**»: «**Queue**».
4. «**GPU Configuration**» (hasta 3 en orden de prioridad). Precios de la documentación (por segundo; × 3600 = por hora):

   | Prioridad | GPU (rótulo de la doc) | VRAM | US$/s | **US$/h** (→ `RUNPOD_TARIFA_GPU_HORA`) |
   |---|---|---|---|---|
   | 1 | «6000s PRO» | 96 GB | 0,00111 | **4,00** |
   | 2 | «H100 PRO» | 80 GB | 0,00116 | **4,18** |
   | 3 | «A100» | 80 GB | 0,00076 | **2,74** |
   | (barata, para probar Wan 5B o LTX) | «L40, L40S, 6000 Ada PRO» | 48 GB | 0,00053 | 1,91 |

   H3 pide 24 GB de VRAM **y 64 GB de RAM** (wiki de FastH3); con 80–96 GB va cómodo. Comprueba el precio real en la consola el día que lo crees.
5. Escalado: «**Active workers**» = **0** (no cobra en reposo), «**Max workers**» = **2** para la prueba, «**Idle timeout**» = **60–120 s** (el valor por defecto es 5 s; más alto evita arranques en frío entre tomas seguidas), «**FlashBoot**» encendido.
6. «**Execution timeout**»: **1500 s** (debe ser mayor que `TIEMPO_MAX_SEG`=1200 del handler).
7. Variables de entorno del endpoint (opcionales): `TIEMPO_MAX_SEG=1200`. *No hace falta `HF_TOKEN` aquí: el endpoint no descarga nada.*
8. Pulsa «**Deploy Endpoint**».
9. Adjunta el volumen: en el endpoint → «**Manage**» → «**Edit Endpoint**» → abre «**Advanced**» → «**Network Volumes**» → elige `series-ia-modelos` → «**Save Endpoint**» ([documentación](https://docs.runpod.io/storage/network-volumes)). En «**Advanced**» también: «**CUDA version selection**» = 12.8 y todas las más nuevas (la imagen usa torch con CUDA 12.8).
10. **Endpoint ID:** es el código que aparece en la página del endpoint y en sus URLs (`https://api.runpod.ai/v2/<ENDPOINT_ID>/run`). *No confirmé en la documentación el rótulo exacto donde se muestra.*

### (g) Qué pegar en la app
| Dato | Dónde |
|---|---|
| `RUNPOD_API_KEY` | En la app: **Estudio → Llaves**, proveedor «**RunPod**» |
| `RUNPOD_ENDPOINT_ID` | Variable de entorno del servidor (Easypanel) |
| `HF_TOKEN` | En la app: **Estudio → Llaves** «**Hugging Face**», y como variable en el pod de descarga (paso d) |
| `RUNPOD_TARIFA_GPU_HORA` | US$/h de la GPU elegida (tabla del paso f), para calcular el costo real con `segundos_gpu` |

### (h) Probar con `curl`
Sin imprimir la llave (Git Bash o Linux; las variables ya cargadas):
```bash
# Asíncrono (recomendado para video)
curl -s -X POST "https://api.runpod.ai/v2/$RUNPOD_ENDPOINT_ID/run" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" -H "Content-Type: application/json" \
  -d @E:/series-ia/gpu/pruebas/entrada-ejemplo.json
# → {"id":"…","status":"IN_QUEUE"}; luego, con ese id:
curl -s "https://api.runpod.ai/v2/$RUNPOD_ENDPOINT_ID/status/EL_ID" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" -o respuesta.json

# Síncrono (solo para tomas cortas; espera poco y luego devuelve el id para seguir con /status)
curl -s -X POST "https://api.runpod.ai/v2/$RUNPOD_ENDPOINT_ID/runsync" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" -H "Content-Type: application/json" \
  -d @E:/series-ia/gpu/pruebas/entrada-ejemplo.json -o respuesta.json

# Guardar el MP4 que vino en base64
python -c "import json,base64; r=json.load(open('respuesta.json')); o=r.get('output',{}); print(o.get('error') or {k:v for k,v in o.items() if k!='video_base64'}); open('prueba.mp4','wb').write(base64.b64decode(o['video_base64'])) if 'video_base64' in o else None"
```
`entrada-ejemplo.json` usa `wan22-ti2v-5b` (el más liviano: no necesita imagen ni licencia).

Para una imagen, cambia el archivo por `entrada-ejemplo-imagen.json` (`zimage-t2i`) y guarda el PNG así:
```bash
python -c "import json,base64; r=json.load(open('respuesta.json')); o=r.get('output',{}); print(o.get('error') or {k:v for k,v in o.items() if k!='imagen_base64'}); open('prueba.png','wb').write(base64.b64decode(o['imagen_base64'])) if 'imagen_base64' in o else None"
``` El primer trabajo tarda más (arranque en frío + carga del modelo desde el volumen).

### (i) Costos de la prueba de 1 día y licencias
De `docs/10-SERVIDOR-GPU-PROPIO.md`, sección 8:

| Paso | Costo estimado |
|---|---|
| Volumen de red de 200 GB (prorrateado) | ~US$0,5 |
| Pod para bajar modelos (~1 h) | ~US$2,9 si es H100; menos con una GPU barata |
| 3 tomas con H3, LTX-2.5 y Wan 2.2 + labios | US$6–9 |
| H3 repetido en RTX 4090 | ~US$0,9 |
| Endpoint serverless: medir arranque en frío (3 trabajos) | US$2–4 |
| **Total** | **~US$13–18** |

RunPod cobra también **la carga del modelo** en cada arranque en frío ([facturación serverless](https://docs.runpod.io/serverless/pricing)). Con «Active workers» = 0 no se paga nada cuando no hay trabajos (solo el volumen, ~US$14/mes).

> **⚠ Licencia de MiniMax H3 (sección 1.4 del documento 10).** Los videos hechos con H3 **no se pueden usar ni mostrar en EE. UU., la Unión Europea, el Reino Unido ni Corea del Sur**. Hay que poner **«MiniMax H3»** visible en la app, avisar que es contenido de IA y, por encima de US$20 M de ingresos al año, pedir permiso a MiniMax (api@minimax.io). Docs.comfy.org agrega que el uso comercial de salidas locales requiere licencia comercial de MiniMax. **Usa `h3-ref` solo para series de Latinoamérica** (o con permiso escrito) y que lo revise un abogado.
>
> LTX-2.5: gratis para empresas con menos de US$10 M de ingresos (licencia comunitaria de Lightricks). Wan 2.2, Z-Image y Qwen-Image-Edit 2511: Apache 2.0.

---

## Para el desarrollador

- **Pruebas sin GPU:** `python -m unittest discover -s E:/series-ia/gpu/pruebas -v` (55 pruebas: frames, dimensiones, marcadores, validación, armado de los 6 flujos con todas las combinaciones, referencias de Qwen en positivo y negativo, y el handler completo con HTTP simulado para video e imagen: base64, PUT, URL pública, nodo faltante, modelo faltante, OOM, rechazo 400, tiempo agotado, salida sin imagen). También `python -m py_compile handler.py`.
- **Cómo arma el flujo:** `armar_prompt_api` quita los nodos con `"_si"` que no se cumple (`"imagen_inicial"`, `"!audio_referencia"`, listas = «cualquiera»), resuelve entradas `{"_si", "entonces", "si_no"}`, reemplaza `{{MARCADOR}}` conservando el tipo, agrega un `LoadImage` por referencia según `_meta.referencias` (H3: nodo 8, `ref_images.ref_image_0..8`; Qwen: nodos 6 y 7, `image1..3`), borra enlaces a nodos quitados y poda lo que no llega a `SaveVideo`/`SaveImage`. Los flujos con `"tipo": "imagen"` en `FLUJOS` no calculan frames y devuelven `imagen_*`.
- **Antes de mandar a ComfyUI** consulta `/object_info` y avisa qué `class_type` faltan o qué archivo de modelo no está (con su carpeta).
- **Variables:** `TIEMPO_MAX_SEG` (1200), `COMFY_ESPERA_ARRANQUE_SEG` (600), `INTERVALO_SONDEO_SEG` (2), `MAX_MB_ARCHIVO_ENTRADA` (60), `COMFY_HOST` (127.0.0.1:8188).
- **Modelos:** `/runpod-volume/models/{diffusion_models,text_encoders,vae,loras,latent_upscale_models}/` (ver `extra_model_paths.yaml`).

### Qué quedó por confirmar en la primera corrida con GPU (TODO)
1. Claves de entradas «DynamicCombo» en formato API: `"format": "mp4", "format.codec": "h264"` (SaveVideo) y `"selection": "vsa", "selection.keep_percent": 10` (BlockSparseAttention). Sale del código de ComfyUI v0.44.2 (`_io.py`), pero no lo vi en un JSON exportado. Si ComfyUI lo rechaza, el error 400 lo dirá con el nodo exacto.
2. Que la subida a ComfyUI v0.44.2 encima de la base 5.10.0 (ComfyUI 0.34.0) arranque bien: la prueba de humo del Dockerfile (`--quick-test-for-ci --cpu`) lo verifica al construir.
3. H3: audio del TTS anclado con `MiniMaxH3AddGuide` (pista desde el fotograma 0) frente a `ref_audios` (solo timbre). Y el texto `subject_definitions` que antepone el handler.
4. LTX-2.5 con `audio_referencia`: se copió el patrón de la plantilla `video_ltx2_3_ia2v` (no hay plantilla ia2v de LTX-2.5).
5. Rendimiento real de int8/nvfp4 (comfy-kitchen) en H100 y A100; VRAM y RAM reales; segundos de GPU por segundo de video.
6. Límite de tamaño de la **salida** de RunPod con `video_base64` (la documentación da 10/20 MB para la entrada). Para clips de más de ~5 MB usa `salida.url_subida`.
7. Rótulos de la consola que no están en la documentación: botón de recarga, nombre de la plantilla del pod, dónde se muestra el «Endpoint ID».
8. Imagen: segundos de GPU por imagen de `zimage-t2i` y `qwen-edit-ref` (40 pasos con cfg 4 en un modelo de 20B puede pasar del minuto; la plantilla sugiere 20 pasos y existe la LoRA Lightning de 4 pasos, que hoy no está en el volumen). En Qwen, comprobar si el prompt respeta más «Picture 1» o «image 1» y si hace falta el `FluxKontextImageScale` de la plantilla sobre la imagen 1.
