"""Trabajador GPU de Series IA para RunPod Serverless.

Recibe el contrato fijo de la app (ver README.md, sección «Contrato»), arma el
flujo de ComfyUI que corresponde (carpeta flujos/), lo manda al ComfyUI local que
arranca la imagen base runpod/worker-comfyui y devuelve el MP4 o, en los flujos
de imagen, el PNG (en base64 o subido con PUT a una URL prefirmada).

Solo usa la librería estándar de Python; `runpod` se importa únicamente al
arrancar como trabajador, así las pruebas corren en cualquier PC sin GPU.

Las funciones puras (frames, dimensiones, marcadores, validación, armado del
flujo) están separadas de las de entrada/salida para poder probarlas.
"""

from __future__ import annotations

import base64
import binascii
import copy
import json
import mimetypes
import os
import random
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

# ---------------------------------------------------------------------------
# Configuración (variables de entorno del endpoint)
# ---------------------------------------------------------------------------
COMFY_HOST = os.environ.get("COMFY_HOST", "127.0.0.1:8188")
DIR_INPUT = os.environ.get("COMFY_INPUT_DIR", "/comfyui/input")
DIR_OUTPUT = os.environ.get("COMFY_OUTPUT_DIR", "/comfyui/output")
DIR_FLUJOS = os.environ.get(
    "FLUJOS_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "flujos")
)
ARCHIVO_PID_COMFY = os.environ.get("COMFY_PID_FILE", "/tmp/comfyui.pid")
TIEMPO_MAX_SEG = float(os.environ.get("TIEMPO_MAX_SEG", "1200"))
ESPERA_ARRANQUE_SEG = float(os.environ.get("COMFY_ESPERA_ARRANQUE_SEG", "600"))
INTERVALO_SONDEO_SEG = float(os.environ.get("INTERVALO_SONDEO_SEG", "2"))
MAX_MB_ARCHIVO_ENTRADA = float(os.environ.get("MAX_MB_ARCHIVO_ENTRADA", "60"))

# Reglas de cada flujo. Fuentes en el "_meta" de cada JSON de flujos/.
#  - tipo: "video" (por defecto) o "imagen" (no usan duracion_seg, fps ni audio)
#  - regla_frames: "4n+1" (Wan), "8n+1" (LTX), "17k+5" (MiniMax H3, a 24 fps)
#  - multiplo: múltiplo al que se redondean ancho y alto
#  - fps_fijo: H3 solo trabaja a 24 fps (constante FPS=24 en nodes_minimax_h3.py)
FLUJOS: dict[str, dict] = {
    "h3-ref": {
        "regla_frames": "17k+5", "fps": 24, "fps_fijo": True, "multiplo": 32,
        "duracion": (4, 15), "dimensiones": (768, 1344),
        "requiere_imagen": False, "requiere_referencias": True, "max_referencias": 9,
        "usa_audio_referencia": True, "tiene_audio": True,
    },
    "ltx25-i2v": {
        # Múltiplo 64: la primera pasada corre a media resolución y necesita múltiplos de 32.
        "regla_frames": "8n+1", "fps": 24, "fps_fijo": False, "multiplo": 64,
        "duracion": (2, 10), "dimensiones": (720, 1280),
        "requiere_imagen": True, "requiere_referencias": False, "max_referencias": 0,
        "usa_audio_referencia": True, "tiene_audio": True,
    },
    "wan22-i2v": {
        "regla_frames": "4n+1", "fps": 16, "fps_fijo": False, "multiplo": 16,
        "duracion": (2, 10), "dimensiones": (720, 1280),
        "requiere_imagen": True, "requiere_referencias": False, "max_referencias": 0,
        "usa_audio_referencia": False, "tiene_audio": False,
    },
    "wan22-ti2v-5b": {
        # La plantilla oficial del 5B usa 24 fps (no 16 como el 14B).
        "regla_frames": "4n+1", "fps": 24, "fps_fijo": False, "multiplo": 32,
        "duracion": (2, 10), "dimensiones": (720, 1280),
        "requiere_imagen": False, "requiere_referencias": False, "max_referencias": 0,
        "usa_audio_referencia": False, "tiene_audio": False,
    },
    # Flujos de imagen: latente de 16 canales con VAE /8 y parches de 2 → lados múltiplo de 16.
    "zimage-t2i": {
        "tipo": "imagen", "multiplo": 16, "dimensiones": (768, 1344),
        "requiere_imagen": False, "requiere_referencias": False, "max_referencias": 0,
    },
    "qwen-edit-ref": {
        "tipo": "imagen", "multiplo": 16, "dimensiones": (768, 1344),
        "requiere_imagen": False, "requiere_referencias": True, "max_referencias": 3,
    },
}


def es_flujo_de_imagen(flujo: str) -> bool:
    return FLUJOS.get(flujo, {}).get("tipo") == "imagen"


# Entradas de nodos cargadores cuyo valor es un archivo de modelo.
ENTRADAS_DE_MODELO = (
    "unet_name", "clip_name", "clip_name1", "clip_name2", "vae_name",
    "lora_name", "model_name", "ckpt_name",
)
# class_type de nodos que producen video o imagen.
NODOS_DE_SALIDA = ("SaveVideo", "VHS_VideoCombine", "SaveWEBM", "SaveAnimatedWEBP", "SaveImage")
EXTENSIONES_VIDEO = (".mp4", ".webm", ".mov", ".mkv")
MIME_POR_EXTENSION_IMAGEN = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}
EXT_POR_MIME = {
    "image/png": ".png", "image/jpeg": ".jpg", "image/jpg": ".jpg", "image/webp": ".webp",
    "audio/wav": ".wav", "audio/x-wav": ".wav", "audio/wave": ".wav", "audio/mpeg": ".mp3",
    "audio/mp3": ".mp3", "audio/ogg": ".ogg", "audio/flac": ".flac", "audio/mp4": ".m4a",
    "audio/x-m4a": ".m4a", "audio/aac": ".aac",
}
PATRON_MARCADOR = re.compile(r"\{\{([A-Z0-9_]+)\}\}")


class ErrorTrabajo(Exception):
    """Error con mensaje en español pensado para mostrarse tal cual en la app."""


# ---------------------------------------------------------------------------
# Funciones puras
# ---------------------------------------------------------------------------
def _redondear(x: float) -> int:
    """Redondeo «normal» (0,5 sube), sin el redondeo bancario de Python."""
    return int(x + 0.5) if x >= 0 else -int(-x + 0.5)


def calcular_frames(duracion_seg: float, fps: int, regla: str) -> int:
    """Cantidad de fotogramas válida para el modelo.

    4n+1  -> Wan 2.2 (81 = 5 s a 16 fps)
    8n+1  -> LTX-2.5 (121 = 5 s a 24 fps)
    17k+5 -> MiniMax H3 a 24 fps (124 = ~5 s); se sube al siguiente valor válido,
             igual que la fórmula de la plantilla oficial de FastH3 OmniRef.
    """
    if duracion_seg <= 0 or fps <= 0:
        raise ErrorTrabajo("La duración y los fps tienen que ser mayores que cero.")
    bruto = duracion_seg * fps
    if regla == "4n+1":
        return 4 * max(1, _redondear(bruto / 4)) + 1
    if regla == "8n+1":
        return 8 * max(1, _redondear(bruto / 8)) + 1
    if regla == "17k+5":
        f = max(5, _redondear(bruto))
        while f % 17 != 5:
            f += 1
        return f
    raise ErrorTrabajo(f"Regla de frames desconocida: {regla}")


def redondear_dimension(valor: int, multiplo: int) -> int:
    """Lleva un lado al múltiplo más cercano; en empate baja (gasta menos VRAM)."""
    if valor <= 0:
        raise ErrorTrabajo("El ancho y el alto tienen que ser mayores que cero.")
    return max(multiplo, ((int(valor) + multiplo // 2 - 1) // multiplo) * multiplo)


def _es_archivo(valor) -> bool:
    return isinstance(valor, dict) and (
        isinstance(valor.get("base64"), str) or isinstance(valor.get("url"), str)
    )


def _validar_archivo(valor, campo: str) -> dict:
    if not _es_archivo(valor):
        raise ErrorTrabajo(
            f"«{campo}» tiene que ser un objeto con «nombre», «mime» y «base64» o «url»."
        )
    url = valor.get("url")
    if url and not isinstance(valor.get("base64"), str):
        if urllib.parse.urlparse(url).scheme not in ("http", "https"):
            raise ErrorTrabajo(f"La URL de «{campo}» tiene que empezar con http:// o https://.")
    return valor


def _numero(valor, campo: str, entero: bool = False):
    if isinstance(valor, bool) or not isinstance(valor, (int, float)):
        raise ErrorTrabajo(f"«{campo}» tiene que ser un número.")
    if entero and int(valor) != valor:
        raise ErrorTrabajo(f"«{campo}» tiene que ser un número entero.")
    return int(valor) if entero else float(valor)


def validar_entrada(entrada) -> dict:
    """Valida el contrato y devuelve una versión normalizada (con valores por defecto)."""
    if not isinstance(entrada, dict):
        raise ErrorTrabajo("Falta el objeto «input» del trabajo.")
    flujo = entrada.get("flujo")
    if flujo not in FLUJOS:
        raise ErrorTrabajo(
            f"Flujo desconocido «{flujo}». Usa uno de: {', '.join(sorted(FLUJOS))}."
        )
    reglas = FLUJOS[flujo]
    es_imagen = es_flujo_de_imagen(flujo)
    prompt = entrada.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ErrorTrabajo("Falta el «prompt» (texto de la imagen)." if es_imagen
                           else "Falta el «prompt» (texto de la toma).")
    negativo = entrada.get("prompt_negativo")
    if negativo is not None and not isinstance(negativo, str):
        raise ErrorTrabajo("«prompt_negativo» tiene que ser texto.")

    # Los flujos de imagen no usan duración, fps ni audio (si llegan, se ignoran).
    duracion = fps = None
    if not es_imagen:
        if "duracion_seg" not in entrada:
            raise ErrorTrabajo("Falta «duracion_seg».")
        duracion = _numero(entrada["duracion_seg"], "duracion_seg")
        dmin, dmax = reglas["duracion"]
        duracion = min(max(duracion, dmin), dmax)  # la app ya redondea; aquí solo se acota

    ancho_def, alto_def = reglas["dimensiones"]
    ancho = _numero(entrada.get("ancho", ancho_def), "ancho", entero=True)
    alto = _numero(entrada.get("alto", alto_def), "alto", entero=True)
    if not (64 <= ancho <= 4096 and 64 <= alto <= 4096):
        raise ErrorTrabajo("«ancho» y «alto» tienen que estar entre 64 y 4096 píxeles.")

    if not es_imagen:
        fps = reglas["fps"]
        if entrada.get("fps") is not None and not reglas["fps_fijo"]:
            fps = _numero(entrada["fps"], "fps", entero=True)
            if not 1 <= fps <= 60:
                raise ErrorTrabajo("«fps» tiene que estar entre 1 y 60.")

    semilla = entrada.get("semilla")
    if semilla is None:
        semilla = random.randint(0, 2**32 - 1)
    else:
        semilla = _numero(semilla, "semilla", entero=True)
        if semilla < 0:
            raise ErrorTrabajo("«semilla» no puede ser negativa.")

    imagen = entrada.get("imagen_inicial")
    if imagen is not None:
        imagen = _validar_archivo(imagen, "imagen_inicial")
    referencias = entrada.get("referencias") or []
    if not isinstance(referencias, list):
        raise ErrorTrabajo("«referencias» tiene que ser una lista.")
    referencias = [_validar_archivo(r, f"referencias[{i}]") for i, r in enumerate(referencias)]
    audio = entrada.get("audio_referencia")
    if audio is not None:
        audio = _validar_archivo(audio, "audio_referencia")

    if reglas["requiere_imagen"] and imagen is None:
        raise ErrorTrabajo(f"El flujo «{flujo}» necesita «imagen_inicial» (primer fotograma).")
    if reglas["requiere_referencias"]:
        if not referencias and imagen is not None:
            referencias = [imagen]  # sin hoja de personaje, la imagen inicial hace de <Picture 1>
        if not referencias:
            raise ErrorTrabajo(
                f"El flujo «{flujo}» necesita al menos una imagen en «referencias» "
                "(hoja de personaje) o una «imagen_inicial»."
            )
        if len(referencias) > reglas["max_referencias"]:
            raise ErrorTrabajo(
                f"El flujo «{flujo}» acepta como máximo {reglas['max_referencias']} referencias "
                f"y llegaron {len(referencias)}."
            )
    else:
        referencias = []  # los demás flujos no usan referencias
    if es_imagen:
        # Los flujos de imagen no tienen primer fotograma: en qwen-edit-ref la
        # imagen_inicial ya pasó a ser la referencia 1; en zimage-t2i se ignora.
        imagen = None
    if not reglas.get("usa_audio_referencia"):
        audio = None

    generar_audio = False
    if not es_imagen:
        generar_audio = entrada.get("generar_audio", True)
        if not isinstance(generar_audio, bool):
            raise ErrorTrabajo("«generar_audio» tiene que ser true o false.")
        generar_audio = generar_audio and reglas["tiene_audio"]

    salida = entrada.get("salida")
    if salida is not None:
        if not isinstance(salida, dict) or not isinstance(salida.get("url_subida"), str):
            raise ErrorTrabajo("«salida» tiene que traer «url_subida» (URL prefirmada para PUT).")
        if urllib.parse.urlparse(salida["url_subida"]).scheme not in ("http", "https"):
            raise ErrorTrabajo("«salida.url_subida» tiene que empezar con http:// o https://.")
        pub = salida.get("url_publica")
        if pub is not None and not isinstance(pub, str):
            raise ErrorTrabajo("«salida.url_publica» tiene que ser texto.")

    return {
        "flujo": flujo,
        "tipo": "imagen" if es_imagen else "video",
        "prompt": prompt.strip(),
        "prompt_negativo": negativo,
        "duracion_seg": duracion,
        "ancho": redondear_dimension(ancho, reglas["multiplo"]),
        "alto": redondear_dimension(alto, reglas["multiplo"]),
        "fps": fps,
        "semilla": semilla,
        "imagen_inicial": imagen,
        "referencias": referencias,
        "audio_referencia": audio,
        "generar_audio": generar_audio,
        "salida": salida,
    }


def preparar_prompt_h3(prompt: str, n_referencias: int) -> str:
    """H3 llama a las imágenes de referencia <Picture 1>, <Picture 2>… (en orden de conexión).

    La app escribe <Subject 1>, <Subject 2>… Si el prompt no menciona ninguna
    <Picture N> ni trae ya «subject_definitions:», se antepone un bloque que
    une cada <Subject i> con su <Picture i>, con el formato de la plantilla
    oficial (video_fasth3_ominiref_r2v.json).
    TODO: afinar con la guía oficial VIDEO_PROMPT_WRITING_GUIDE_ref_en.md cuando
    se pruebe en GPU (la app puede mandar el prompt completo y esto no se toca).
    """
    if n_referencias <= 0 or "<Picture" in prompt or "subject_definitions:" in prompt:
        return prompt
    lineas = ["subject_definitions:"]
    for i in range(1, n_referencias + 1):
        lineas.append(f"<Subject {i}> is the character shown in <Picture {i}>.")
    return "\n".join(lineas) + "\n\ndetailed_description:\n" + prompt


def condicion_cumplida(condicion, banderas: dict) -> bool:
    """"bandera", "!bandera" o lista (basta con que se cumpla una)."""
    if isinstance(condicion, list):
        return any(condicion_cumplida(c, banderas) for c in condicion)
    if not isinstance(condicion, str):
        raise ErrorTrabajo(f"Condición inválida en el flujo: {condicion!r}")
    if condicion.startswith("!"):
        return not bool(banderas.get(condicion[1:]))
    return bool(banderas.get(condicion))


def _resolver_condicional(valor, banderas: dict):
    while isinstance(valor, dict) and "_si" in valor:
        valor = valor.get("entonces") if condicion_cumplida(valor["_si"], banderas) else valor.get("si_no")
    return valor


def reemplazar_marcadores(objeto, valores: dict):
    """Reemplaza "{{MARCADOR}}" en todo el objeto.

    Si la cadena es solo el marcador, se pone el valor con su tipo (número,
    booleano…); si el marcador va dentro de un texto, se inserta como texto.
    """
    if isinstance(objeto, dict):
        return {k: reemplazar_marcadores(v, valores) for k, v in objeto.items()}
    if isinstance(objeto, list):
        return [reemplazar_marcadores(v, valores) for v in objeto]
    if isinstance(objeto, str):
        completo = PATRON_MARCADOR.fullmatch(objeto)
        if completo:
            nombre = completo.group(1)
            if nombre not in valores:
                raise ErrorTrabajo(f"El flujo usa el marcador {{{{{nombre}}}}} y no tiene valor.")
            return copy.deepcopy(valores[nombre])

        def _sustituir(m):
            if m.group(1) not in valores:
                raise ErrorTrabajo(f"El flujo usa el marcador {{{{{m.group(1)}}}}} y no tiene valor.")
            return str(valores[m.group(1)])

        return PATRON_MARCADOR.sub(_sustituir, objeto)
    return objeto


def _es_enlace(valor) -> bool:
    return (
        isinstance(valor, list) and len(valor) == 2
        and isinstance(valor[0], str) and isinstance(valor[1], int)
    )


def armar_prompt_api(plantilla: dict, valores: dict, banderas: dict,
                     nombres_referencias: list[str] | None = None) -> dict:
    """Convierte la plantilla de flujos/ en el «prompt» de la API de ComfyUI.

    1. Quita los nodos cuyo "_si" no se cumple.
    2. Resuelve las entradas condicionales {"_si": …, "entonces": …, "si_no": …}.
    3. Reemplaza los marcadores {{…}}.
    4. Agrega un LoadImage por cada referencia y lo conecta al nodo indicado en _meta.
    5. Quita entradas que apuntan a nodos que ya no existen (entradas opcionales).
    6. Poda los nodos que no llevan a un nodo de salida.
    """
    plantilla = copy.deepcopy(plantilla)
    meta = plantilla.pop("_meta", {}) or {}
    nodos = {}
    for nid, nodo in plantilla.items():
        if not isinstance(nodo, dict) or "class_type" not in nodo:
            continue
        if "_si" in nodo and not condicion_cumplida(nodo["_si"], banderas):
            continue
        nodos[nid] = nodo

    for nodo in nodos.values():
        entradas = {}
        for k, v in (nodo.get("inputs") or {}).items():
            v = _resolver_condicional(v, banderas)
            if v is not None:
                entradas[k] = v
        nodo["inputs"] = reemplazar_marcadores(entradas, valores)

    # "_meta.referencias": {"nodo": "8"} o {"nodos": ["6", "7"]} (la misma imagen va a
    # todos), "prefijo" de la entrada y "inicio" del índice (0 en H3, 1 en image1..3 de Qwen).
    conf_refs = meta.get("referencias")
    if nombres_referencias:
        ids_destino = []
        if conf_refs:
            ids_destino = conf_refs.get("nodos") or [conf_refs.get("nodo")]
        if not ids_destino or any(d not in nodos for d in ids_destino):
            raise ErrorTrabajo("Este flujo no acepta imágenes de referencia.")
        maximo = conf_refs.get("maximo")
        if maximo is not None and len(nombres_referencias) > maximo:
            raise ErrorTrabajo(
                f"Este flujo acepta como máximo {maximo} referencias y llegaron {len(nombres_referencias)}."
            )
        inicio = int(conf_refs.get("inicio", 0))
        for i, nombre in enumerate(nombres_referencias):
            nid = f"ref_{i + 1}"
            nodos[nid] = {
                "class_type": "LoadImage",
                "inputs": {"image": nombre},
                "_meta": {"title": f"Referencia {i + 1} (<Picture {i + 1}>)"},
            }
            for d in ids_destino:
                nodos[d]["inputs"][f"{conf_refs['prefijo']}{inicio + i}"] = [nid, 0]

    # Entradas que apuntan a nodos quitados (p. ej. sin imagen inicial).
    cambio = True
    while cambio:
        cambio = False
        for nodo in nodos.values():
            for k, v in list(nodo["inputs"].items()):
                if _es_enlace(v) and v[0] not in nodos:
                    del nodo["inputs"][k]
                    cambio = True

    # Poda: solo lo que alimenta a un nodo de salida.
    salidas = [nid for nid, n in nodos.items() if n["class_type"] in NODOS_DE_SALIDA]
    if not salidas:
        raise ErrorTrabajo("El flujo no tiene nodo de salida (SaveVideo, VHS_VideoCombine o SaveImage).")
    vivos, pendientes = set(), list(salidas)
    while pendientes:
        nid = pendientes.pop()
        if nid in vivos:
            continue
        vivos.add(nid)
        for v in nodos[nid]["inputs"].values():
            if _es_enlace(v) and v[0] in nodos:
                pendientes.append(v[0])

    prompt = {}
    for nid in nodos:
        if nid not in vivos:
            continue
        nodo = nodos[nid]
        limpio = {"class_type": nodo["class_type"], "inputs": nodo["inputs"]}
        if isinstance(nodo.get("_meta"), dict):
            limpio["_meta"] = nodo["_meta"]  # ComfyUI acepta "_meta" (título) dentro del nodo
        prompt[nid] = limpio
    return prompt


def clases_faltantes(prompt: dict, object_info: dict) -> list[str]:
    return sorted({n["class_type"] for n in prompt.values() if n["class_type"] not in object_info})


def _opciones_de_entrada(spec) -> list | None:
    """Opciones de un combo en /object_info (formato viejo y formato V3 "COMBO")."""
    if not isinstance(spec, (list, tuple)) or not spec:
        return None
    if isinstance(spec[0], list):
        return spec[0]
    if spec[0] == "COMBO" and len(spec) > 1 and isinstance(spec[1], dict):
        opciones = spec[1].get("options")
        return opciones if isinstance(opciones, list) else None
    return None


def modelos_faltantes(prompt: dict, object_info: dict, carpetas: dict | None = None) -> list[str]:
    """Archivos de modelo que pide el flujo y ComfyUI no ve en sus carpetas."""
    carpetas = carpetas or {}
    faltan = []
    for nodo in prompt.values():
        info = object_info.get(nodo["class_type"]) or {}
        entradas_info = info.get("input") or {}
        for nombre_entrada in ENTRADAS_DE_MODELO:
            valor = nodo["inputs"].get(nombre_entrada)
            if not isinstance(valor, str):
                continue
            spec = (entradas_info.get("required") or {}).get(nombre_entrada) or \
                   (entradas_info.get("optional") or {}).get(nombre_entrada)
            opciones = _opciones_de_entrada(spec)
            if opciones is not None and valor not in opciones:
                carpeta = carpetas.get(valor, "?")
                faltan.append(f"{valor} (carpeta models/{carpeta}/)")
    return sorted(set(faltan))


def extraer_videos(historial: dict) -> list[dict]:
    """Busca los videos en la salida de /history (SaveVideo usa "images"; VHS usa "gifs")."""
    videos = []
    for salida in (historial.get("outputs") or {}).values():
        if not isinstance(salida, dict):
            continue
        for clave in ("gifs", "videos", "video", "images"):
            for item in salida.get(clave) or []:
                if isinstance(item, dict) and str(item.get("filename", "")).lower().endswith(EXTENSIONES_VIDEO):
                    videos.append(item)
    return videos


def extraer_imagenes(historial: dict) -> list[dict]:
    """Busca las imágenes en la salida de /history (SaveImage usa "images" con type "output").

    Primero las de tipo "output"; las "temp" (PreviewImage) solo si no hay otras.
    """
    guardadas, temporales = [], []
    for salida in (historial.get("outputs") or {}).values():
        if not isinstance(salida, dict):
            continue
        for item in salida.get("images") or []:
            if not isinstance(item, dict):
                continue
            ext = os.path.splitext(str(item.get("filename", "")).lower())[1]
            if ext not in MIME_POR_EXTENSION_IMAGEN:
                continue
            (guardadas if item.get("type", "output") == "output" else temporales).append(item)
    return guardadas or temporales


def mime_de_imagen(nombre: str) -> str:
    return MIME_POR_EXTENSION_IMAGEN.get(os.path.splitext(nombre.lower())[1], "image/png")


def es_error_de_memoria(texto: str) -> bool:
    t = (texto or "").lower()
    return "out of memory" in t or "outofmemory" in t or "cuda oom" in t or "allocation on device" in t


def describir_error_ejecucion(estado: dict) -> str:
    """Mensaje en español a partir del "status" de /history cuando ComfyUI falla."""
    for evento, datos in estado.get("messages") or []:
        if evento != "execution_error" or not isinstance(datos, dict):
            continue
        texto = str(datos.get("exception_message", "")).strip()
        tipo = str(datos.get("exception_type", ""))
        nodo = f"{datos.get('node_type', '?')} (nodo {datos.get('node_id', '?')})"
        if es_error_de_memoria(texto) or es_error_de_memoria(tipo):
            return (
                f"La GPU se quedó sin memoria (OOM de CUDA) en {nodo}. Baja la duración o la "
                "resolución, o usa una GPU con más VRAM."
            )
        return f"ComfyUI falló en {nodo}: {tipo}: {texto}"[:1500]
    return "ComfyUI terminó con error y no dio detalles."


def describir_error_validacion(cuerpo: dict) -> str:
    """Mensaje en español para la respuesta 400 de POST /prompt."""
    partes = []
    err = cuerpo.get("error")
    if isinstance(err, dict):
        partes.append(f"{err.get('message', '')} {err.get('details', '')}".strip())
    elif err:
        partes.append(str(err))
    for nid, info in (cuerpo.get("node_errors") or {}).items():
        for e in (info or {}).get("errors") or []:
            partes.append(f"nodo {nid} ({info.get('class_type', '?')}): {e.get('message', '')} {e.get('details', '')}".strip())
    return "ComfyUI rechazó el flujo: " + " | ".join(p for p in partes if p)[:1500]


def url_sin_query(url: str) -> str:
    p = urllib.parse.urlsplit(url)
    return urllib.parse.urlunsplit((p.scheme, p.netloc, p.path, "", ""))


def nombre_seguro(nombre: str | None, mime: str | None, por_defecto: str) -> str:
    base = os.path.basename(nombre or "") or por_defecto
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", base)[:80]
    raiz, ext = os.path.splitext(base)
    if not ext:
        ext = EXT_POR_MIME.get((mime or "").lower()) or mimetypes.guess_extension(mime or "") or ""
    return (raiz or por_defecto) + ext.lower()


def cargar_flujo(nombre: str) -> dict:
    ruta = os.path.join(DIR_FLUJOS, f"{nombre}.json")
    if not os.path.isfile(ruta):
        raise ErrorTrabajo(f"No encuentro el archivo del flujo: {ruta}")
    with open(ruta, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Entrada/salida (HTTP con ComfyUI y descargas). Todo pasa por `http` para poder simularlo.
# ---------------------------------------------------------------------------
def http(metodo: str, url: str, datos: bytes | None = None,
         cabeceras: dict | None = None, timeout: float = 60) -> tuple[int, bytes]:
    req = urllib.request.Request(url, data=datos, method=metodo, headers=cabeceras or {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read() or b""


def _url_comfy(ruta: str) -> str:
    return f"http://{COMFY_HOST}{ruta}"


def _json(cuerpo: bytes):
    try:
        return json.loads(cuerpo.decode("utf-8") or "null")
    except (ValueError, UnicodeDecodeError):
        return None


def obtener_bytes(archivo: dict, campo: str) -> bytes:
    limite = int(MAX_MB_ARCHIVO_ENTRADA * 1024 * 1024)
    if isinstance(archivo.get("base64"), str):
        datos = archivo["base64"]
        if datos.startswith("data:") and "," in datos:
            datos = datos.split(",", 1)[1]
        try:
            contenido = base64.b64decode(datos, validate=False)
        except (binascii.Error, ValueError):
            raise ErrorTrabajo(f"El base64 de «{campo}» no es válido.")
    else:
        try:
            estado, contenido = http("GET", archivo["url"], timeout=120)
        except Exception as e:  # red caída, DNS, etc.
            raise ErrorTrabajo(f"No pude descargar «{campo}» desde su URL: {e}")
        if estado != 200:
            raise ErrorTrabajo(f"No pude descargar «{campo}» desde su URL (HTTP {estado}).")
    if not contenido:
        raise ErrorTrabajo(f"«{campo}» llegó vacío.")
    if len(contenido) > limite:
        raise ErrorTrabajo(f"«{campo}» pesa más de {MAX_MB_ARCHIVO_ENTRADA:.0f} MB.")
    return contenido


def guardar_en_input(archivo: dict, campo: str, prefijo: str, por_defecto: str) -> str:
    contenido = obtener_bytes(archivo, campo)
    nombre = f"{prefijo}_{uuid.uuid4().hex[:8]}_" + nombre_seguro(
        archivo.get("nombre"), archivo.get("mime"), por_defecto
    )
    os.makedirs(DIR_INPUT, exist_ok=True)
    with open(os.path.join(DIR_INPUT, nombre), "wb") as f:
        f.write(contenido)
    return nombre


def _comfy_sigue_vivo() -> bool:
    try:
        with open(ARCHIVO_PID_COMFY) as f:
            pid = int(f.read().strip())
    except (OSError, ValueError):
        return True  # sin archivo PID no se puede saber; se asume vivo
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except OSError:
        return True


def esperar_comfy(limite_seg: float = None) -> None:
    limite_seg = ESPERA_ARRANQUE_SEG if limite_seg is None else limite_seg
    fin = time.monotonic() + limite_seg
    while True:
        try:
            estado, _ = http("GET", _url_comfy("/"), timeout=5)
            if estado == 200:
                return
        except Exception:
            pass
        if not _comfy_sigue_vivo():
            raise ErrorTrabajo("ComfyUI no arrancó (el proceso murió). Revisa los registros del trabajador.")
        if time.monotonic() >= fin:
            raise ErrorTrabajo(f"ComfyUI no respondió en {limite_seg:.0f} s después de arrancar.")
        time.sleep(1)


_CACHE_OBJECT_INFO: dict | None = None


def obtener_object_info() -> dict:
    global _CACHE_OBJECT_INFO
    if _CACHE_OBJECT_INFO is None:
        estado, cuerpo = http("GET", _url_comfy("/object_info"), timeout=120)
        datos = _json(cuerpo)
        if estado != 200 or not isinstance(datos, dict):
            raise ErrorTrabajo(f"No pude leer la lista de nodos de ComfyUI (/object_info, HTTP {estado}).")
        _CACHE_OBJECT_INFO = datos
    return _CACHE_OBJECT_INFO


def enviar_prompt(prompt: dict, id_cliente: str) -> str:
    cuerpo = json.dumps({"prompt": prompt, "client_id": id_cliente}).encode("utf-8")
    estado, resp = http("POST", _url_comfy("/prompt"), cuerpo, {"Content-Type": "application/json"}, timeout=60)
    datos = _json(resp) or {}
    if estado != 200 or "prompt_id" not in datos:
        if isinstance(datos, dict) and (datos.get("error") or datos.get("node_errors")):
            raise ErrorTrabajo(describir_error_validacion(datos))
        raise ErrorTrabajo(f"ComfyUI no aceptó el flujo (HTTP {estado}).")
    return datos["prompt_id"]


def esperar_resultado(prompt_id: str, limite_seg: float) -> dict:
    fin = time.monotonic() + limite_seg
    fallos_seguidos = 0
    while True:
        try:
            estado, cuerpo = http("GET", _url_comfy(f"/history/{prompt_id}"), timeout=30)
            fallos_seguidos = 0
        except Exception:
            estado, cuerpo = 0, b""
            fallos_seguidos += 1
            if fallos_seguidos >= 10 or not _comfy_sigue_vivo():
                raise ErrorTrabajo(
                    "ComfyUI dejó de responder mientras generaba (posible caída por falta de "
                    "memoria de la GPU o de la RAM)."
                )
        datos = _json(cuerpo) if estado == 200 else None
        if isinstance(datos, dict) and prompt_id in datos:
            hist = datos[prompt_id]
            st = hist.get("status") or {}
            if st.get("status_str") == "error":
                raise ErrorTrabajo(describir_error_ejecucion(st))
            if st.get("completed") or st.get("status_str") == "success" or hist.get("outputs"):
                return hist
        if time.monotonic() >= fin:
            try:
                http("POST", _url_comfy("/interrupt"), b"{}", {"Content-Type": "application/json"}, timeout=10)
            except Exception:
                pass
            raise ErrorTrabajo(
                f"Se agotó el tiempo: la toma no terminó en {limite_seg:.0f} s (TIEMPO_MAX_SEG)."
            )
        time.sleep(INTERVALO_SONDEO_SEG)


def leer_video(item: dict, que: str = "el video") -> bytes:
    """Lee el archivo de salida (video o imagen) del disco o, si no está, por /view."""
    if item.get("type", "output") == "output":
        ruta = os.path.join(DIR_OUTPUT, item.get("subfolder") or "", item["filename"])
        if os.path.isfile(ruta):
            with open(ruta, "rb") as f:
                return f.read()
    consulta = urllib.parse.urlencode({
        "filename": item["filename"], "subfolder": item.get("subfolder") or "",
        "type": item.get("type") or "output",
    })
    estado, cuerpo = http("GET", _url_comfy(f"/view?{consulta}"), timeout=120)
    if estado != 200 or not cuerpo:
        raise ErrorTrabajo(f"No pude leer {que} de salida «{item['filename']}» (HTTP {estado}).")
    return cuerpo


def subir_video(contenido: bytes, salida: dict, mime: str = "video/mp4", que: str = "el video") -> str:
    """Sube el resultado (video o imagen) con PUT a la URL prefirmada."""
    try:
        estado, _ = http("PUT", salida["url_subida"], contenido, {"Content-Type": mime}, timeout=300)
    except Exception as e:
        raise ErrorTrabajo(f"No pude subir {que} a «salida.url_subida»: {e}")
    if estado not in (200, 201, 204):
        raise ErrorTrabajo(f"No pude subir {que} a «salida.url_subida» (HTTP {estado}).")
    return salida.get("url_publica") or url_sin_query(salida["url_subida"])


def _borrar(rutas: list[str]) -> None:
    for ruta in rutas:
        try:
            os.remove(ruta)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Handler de RunPod
# ---------------------------------------------------------------------------
def procesar(entrada: dict, id_trabajo: str) -> dict:
    datos = validar_entrada(entrada)
    flujo = datos["flujo"]
    reglas = FLUJOS[flujo]
    plantilla = cargar_flujo(flujo)
    meta = plantilla.get("_meta") or {}
    es_imagen = datos["tipo"] == "imagen"

    frames = None
    if not es_imagen:
        frames = calcular_frames(datos["duracion_seg"], datos["fps"], reglas["regla_frames"])
    prefijo = re.sub(r"[^A-Za-z0-9_-]", "_", str(id_trabajo))[:40] or "trabajo"

    esperar_comfy()
    object_info = obtener_object_info()

    archivos_creados: list[str] = []
    try:
        nombre_imagen = nombre_audio = None
        if datos["imagen_inicial"] is not None:
            nombre_imagen = guardar_en_input(datos["imagen_inicial"], "imagen_inicial", prefijo, "imagen.png")
            archivos_creados.append(os.path.join(DIR_INPUT, nombre_imagen))
        nombres_refs = []
        for i, ref in enumerate(datos["referencias"]):
            if ref is datos["imagen_inicial"] and nombre_imagen:
                nombres_refs.append(nombre_imagen)
                continue
            n = guardar_en_input(ref, f"referencias[{i}]", prefijo, f"referencia{i + 1}.png")
            archivos_creados.append(os.path.join(DIR_INPUT, n))
            nombres_refs.append(n)
        if datos["audio_referencia"] is not None:
            nombre_audio = guardar_en_input(datos["audio_referencia"], "audio_referencia", prefijo, "voz.wav")
            archivos_creados.append(os.path.join(DIR_INPUT, nombre_audio))

        prompt_texto = datos["prompt"]
        if flujo == "h3-ref":
            prompt_texto = preparar_prompt_h3(prompt_texto, len(nombres_refs))
        negativo = datos["prompt_negativo"]
        if negativo is None:
            negativo = meta.get("prompt_negativo_defecto", "")

        valores = {
            "PROMPT": prompt_texto,
            "PROMPT_NEGATIVO": negativo,
            "ANCHO": datos["ancho"],
            "ALTO": datos["alto"],
            "ANCHO_MEDIO": datos["ancho"] // 2,
            "ALTO_MEDIO": datos["alto"] // 2,
            "SEMILLA": datos["semilla"],
            "IMAGEN_INICIAL": nombre_imagen or "",
            "AUDIO_REFERENCIA": nombre_audio or "",
            "PREFIJO_SALIDA": f"seriesia/{prefijo}",
        }
        if not es_imagen:
            valores.update({
                "FRAMES": frames,
                "FPS": datos["fps"],
                "DURACION_SEG": round(frames / datos["fps"], 4),
            })
        banderas = {
            "imagen_inicial": nombre_imagen is not None,
            "audio_referencia": nombre_audio is not None,
            "generar_audio": datos["generar_audio"],
            "referencias": bool(nombres_refs),
        }
        prompt = armar_prompt_api(plantilla, valores, banderas, nombres_refs)

        faltan = clases_faltantes(prompt, object_info)
        if faltan:
            raise ErrorTrabajo(
                "Faltan nodos de ComfyUI en la imagen Docker (no están instalados o no cargaron): "
                + ", ".join(faltan) + ". Revisa el Dockerfile y los registros de arranque."
            )
        faltan = modelos_faltantes(prompt, object_info, meta.get("carpetas_de_modelos"))
        if faltan:
            raise ErrorTrabajo(
                "Faltan modelos en el volumen de red (/runpod-volume/models/…): " + "; ".join(faltan)
                + ". Córrelos con descargar-modelos.sh."
            )

        prompt_id = enviar_prompt(prompt, id_cliente=prefijo)
        historial = esperar_resultado(prompt_id, TIEMPO_MAX_SEG)
        if es_imagen:
            return _respuesta_imagen(historial, datos, archivos_creados)
        videos = extraer_videos(historial)
        if not videos:
            raise ErrorTrabajo("ComfyUI terminó pero no dejó ningún video (.mp4) en la salida.")
        item = videos[0]
        contenido = leer_video(item)
        if item.get("type", "output") == "output":
            archivos_creados.append(os.path.join(DIR_OUTPUT, item.get("subfolder") or "", item["filename"]))

        respuesta = {
            "mime": "video/mp4",
            "flujo": flujo,
            "ancho": datos["ancho"],
            "alto": datos["alto"],
            "duracion_seg": round(frames / datos["fps"], 3),
            "fps": datos["fps"],
            "frames": frames,
            "semilla": datos["semilla"],
        }
        if datos["salida"]:
            respuesta["video_url"] = subir_video(contenido, datos["salida"])
        else:
            respuesta["video_base64"] = base64.b64encode(contenido).decode("ascii")
        return respuesta
    finally:
        _borrar(archivos_creados)


def _respuesta_imagen(historial: dict, datos: dict, archivos_creados: list[str]) -> dict:
    """Arma la salida de los flujos de imagen: {"imagen_base64"|"imagen_url", "mime", …}."""
    imagenes = extraer_imagenes(historial)
    if not imagenes:
        raise ErrorTrabajo("ComfyUI terminó pero no dejó ninguna imagen (.png) en la salida.")
    item = imagenes[0]
    contenido = leer_video(item, "la imagen")
    if item.get("type", "output") == "output":
        archivos_creados.append(os.path.join(DIR_OUTPUT, item.get("subfolder") or "", item["filename"]))
    mime = mime_de_imagen(item["filename"])
    respuesta = {
        "mime": mime,
        "flujo": datos["flujo"],
        "ancho": datos["ancho"],
        "alto": datos["alto"],
        "semilla": datos["semilla"],
    }
    if datos["salida"]:
        respuesta["imagen_url"] = subir_video(contenido, datos["salida"], mime, "la imagen")
    else:
        respuesta["imagen_base64"] = base64.b64encode(contenido).decode("ascii")
    return respuesta


def handler(job: dict) -> dict:
    inicio = time.monotonic()
    try:
        resultado = procesar((job or {}).get("input"), (job or {}).get("id") or uuid.uuid4().hex)
    except ErrorTrabajo as e:
        return {"error": str(e)}
    except Exception as e:  # cualquier otra cosa: mensaje claro, sin traza larga
        texto = f"{type(e).__name__}: {e}"
        if es_error_de_memoria(texto):
            return {"error": "La GPU se quedó sin memoria (OOM de CUDA). Baja la duración o la resolución."}
        return {"error": f"Error inesperado del trabajador: {texto}"[:1500]}
    resultado["segundos_gpu"] = round(time.monotonic() - inicio, 2)
    return resultado


if __name__ == "__main__":
    import runpod  # viene en la imagen base runpod/worker-comfyui

    runpod.serverless.start({"handler": handler})
