"""Pruebas del handler sin GPU ni ComfyUI: se simulan todas las llamadas HTTP.

Correr:  python -m unittest discover -s E:/series-ia/gpu/pruebas -v
"""

import base64
import json
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)

import handler as h  # noqa: E402

PNG_1PX = base64.b64encode(
    bytes.fromhex(
        "89504e470d0a1a0a0000000d4948445200000001000000010806000000"
        "1f15c4890000000d49444154789c6360000002000154a24f5d0000000049454e44ae426082"
    )
).decode()
WAV = base64.b64encode(b"RIFF\x24\x00\x00\x00WAVEfmt ").decode()
MP4 = b"\x00\x00\x00\x18ftypmp42video-falso"
FLUJOS_VIDEO = ("h3-ref", "ltx25-i2v", "wan22-i2v", "wan22-ti2v-5b")
FLUJOS_CATALOGO = ("catalogo-prenda", "catalogo-modelo", "catalogo-angulos", "mejorar-2k")
FLUJOS_IMAGEN = ("zimage-t2i", "qwen-edit-ref", "qwen-edit-rapido") + FLUJOS_CATALOGO
FLUJOS = FLUJOS_VIDEO + FLUJOS_IMAGEN


def imagen(nombre="foto.png"):
    return {"nombre": nombre, "mime": "image/png", "base64": PNG_1PX}


def cargar(nombre):
    with open(os.path.join(RAIZ, "flujos", f"{nombre}.json"), encoding="utf-8") as f:
        return json.load(f)


def object_info_completo():
    """/object_info falso con todas las clases de los flujos y sus modelos."""
    info = {"LoadImage": {"input": {"required": {"image": [[]]}}}}
    for nombre in FLUJOS:
        plantilla = cargar(nombre)
        for nid, nodo in plantilla.items():
            if nid == "_meta":
                continue
            clase = nodo["class_type"]
            info.setdefault(clase, {"input": {"required": {}}})
            for k, v in nodo["inputs"].items():
                if k in h.ENTRADAS_DE_MODELO:
                    req = info[clase]["input"]["required"]
                    # mitad con el formato viejo y mitad con el formato V3 "COMBO"
                    if clase in ("UNETLoader", "LatentUpscaleModelLoader"):
                        opciones = req.setdefault(k, ["COMBO", {"options": []}])[1]["options"]
                    else:
                        opciones = req.setdefault(k, [[]])[0]
                    if v not in opciones:
                        opciones.append(v)
    return info


class ComfyFalso:
    """Simula ComfyUI y los servidores externos a través de handler.http."""

    def __init__(self, object_info=None, historial=None, estado_prompt=200, cuerpo_prompt=None,
                 estado_put=200, nunca_termina=False):
        self.object_info = object_info if object_info is not None else object_info_completo()
        self.historial = historial
        self.estado_prompt = estado_prompt
        self.cuerpo_prompt = cuerpo_prompt
        self.estado_put = estado_put
        self.nunca_termina = nunca_termina
        self.prompt_enviado = None
        self.puts = []
        self.interrumpido = False

    def __call__(self, metodo, url, datos=None, cabeceras=None, timeout=60):
        if url.endswith("127.0.0.1:8188/"):
            return 200, b"ok"
        if url.endswith("/object_info"):
            return 200, json.dumps(self.object_info).encode()
        if url.endswith("/prompt") and metodo == "POST":
            self.prompt_enviado = json.loads(datos)["prompt"]
            cuerpo = self.cuerpo_prompt or {"prompt_id": "p-123", "number": 1, "node_errors": {}}
            return self.estado_prompt, json.dumps(cuerpo).encode()
        if "/history/" in url:
            if self.nunca_termina:
                return 200, b"{}"
            return 200, json.dumps({"p-123": self.historial}).encode()
        if url.endswith("/interrupt"):
            self.interrumpido = True
            return 200, b""
        if "/view?" in url:
            return 200, (base64.b64decode(PNG_1PX) if ".png" in url else MP4)
        if metodo == "PUT":
            self.puts.append((url, datos, cabeceras))
            return self.estado_put, b""
        if url.startswith("https://ejemplo.com/"):
            return 200, base64.b64decode(PNG_1PX)
        raise AssertionError(f"Llamada HTTP no esperada: {metodo} {url}")


def historial_imagen(archivo="x_00001_.png", tipo="output"):
    return {
        "status": {"status_str": "success", "completed": True, "messages": []},
        "outputs": {"13": {"images": [{"filename": archivo, "subfolder": "seriesia", "type": tipo}]}},
    }


def historial_ok(clave="images", archivo="seriesia/x_00001.mp4"):
    return {
        "status": {"status_str": "success", "completed": True, "messages": []},
        "outputs": {"20": {clave: [{"filename": os.path.basename(archivo), "subfolder": "seriesia", "type": "output"}],
                           "animated": [True]}},
    }


class BaseConCarpetas(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.parches = [
            mock.patch.object(h, "DIR_INPUT", os.path.join(self.tmp, "input")),
            mock.patch.object(h, "DIR_OUTPUT", os.path.join(self.tmp, "output")),
            mock.patch.object(h, "ARCHIVO_PID_COMFY", os.path.join(self.tmp, "no-existe.pid")),
            mock.patch.object(h, "INTERVALO_SONDEO_SEG", 0),
            mock.patch.object(h, "_CACHE_OBJECT_INFO", None),
            mock.patch("time.sleep", lambda s: None),
        ]
        for p in self.parches:
            p.start()

    def tearDown(self):
        for p in self.parches:
            p.stop()
        shutil.rmtree(self.tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
class PruebasFuncionesPuras(unittest.TestCase):
    def test_frames_wan_4n1(self):
        self.assertEqual(h.calcular_frames(5, 16, "4n+1"), 81)
        self.assertEqual(h.calcular_frames(5, 24, "4n+1"), 121)
        self.assertEqual(h.calcular_frames(2, 16, "4n+1"), 33)
        for d in range(2, 11):
            self.assertEqual((h.calcular_frames(d, 16, "4n+1") - 1) % 4, 0)

    def test_frames_ltx_8n1(self):
        self.assertEqual(h.calcular_frames(5, 24, "8n+1"), 121)
        self.assertEqual(h.calcular_frames(10, 24, "8n+1"), 241)
        for d in range(2, 11):
            for fps in (24, 25):
                self.assertEqual((h.calcular_frames(d, fps, "8n+1") - 1) % 8, 0)

    def test_frames_h3_17k5(self):
        # Igual a la fórmula de la plantilla oficial: max(5, round(a*24)) + (5 - x % 17) % 17
        for d in range(4, 16):
            x = max(5, round(d * 24))
            esperado = x + (5 - (x % 17)) % 17
            self.assertEqual(h.calcular_frames(d, 24, "17k+5"), esperado)
        self.assertEqual(h.calcular_frames(5, 24, "17k+5"), 124)
        self.assertEqual(h.calcular_frames(15, 24, "17k+5"), 362)

    def test_frames_errores(self):
        with self.assertRaises(h.ErrorTrabajo):
            h.calcular_frames(0, 24, "4n+1")
        with self.assertRaises(h.ErrorTrabajo):
            h.calcular_frames(5, 24, "rara")

    def test_redondear_dimension(self):
        self.assertEqual(h.redondear_dimension(720, 16), 720)
        self.assertEqual(h.redondear_dimension(720, 32), 704)
        self.assertEqual(h.redondear_dimension(720, 64), 704)
        self.assertEqual(h.redondear_dimension(1280, 64), 1280)
        self.assertEqual(h.redondear_dimension(1344, 32), 1344)
        self.assertEqual(h.redondear_dimension(730, 32), 736)
        self.assertEqual(h.redondear_dimension(10, 32), 32)

    def test_marcadores_conservan_tipos(self):
        plantilla = {"a": "{{ANCHO}}", "b": "{{FPS}}", "c": "texto {{ANCHO}}px", "d": ["{{SEMILLA}}", 1],
                     "e": "{{BANDERA}}", "f": 3.5}
        r = h.reemplazar_marcadores(plantilla, {"ANCHO": 720, "FPS": 24, "SEMILLA": 7, "BANDERA": False})
        self.assertEqual(r, {"a": 720, "b": 24, "c": "texto 720px", "d": [7, 1], "e": False, "f": 3.5})
        self.assertIsInstance(r["a"], int)

    def test_marcador_sin_valor_da_error(self):
        with self.assertRaises(h.ErrorTrabajo):
            h.reemplazar_marcadores({"a": "{{NO_EXISTE}}"}, {})

    def test_condiciones(self):
        b = {"imagen_inicial": True, "audio_referencia": False}
        self.assertTrue(h.condicion_cumplida("imagen_inicial", b))
        self.assertTrue(h.condicion_cumplida("!audio_referencia", b))
        self.assertTrue(h.condicion_cumplida(["audio_referencia", "imagen_inicial"], b))
        self.assertFalse(h.condicion_cumplida(["audio_referencia"], b))

    def test_prompt_h3(self):
        p = h.preparar_prompt_h3("<Subject 1> camina.", 2)
        self.assertIn("<Subject 1> is the character shown in <Picture 1>.", p)
        self.assertIn("<Subject 2> is the character shown in <Picture 2>.", p)
        self.assertTrue(p.endswith("<Subject 1> camina."))
        ya_listo = "subject_definitions:\n<Subject 1> is the woman in <Picture 1>."
        self.assertEqual(h.preparar_prompt_h3(ya_listo, 1), ya_listo)

    def test_url_sin_query(self):
        self.assertEqual(h.url_sin_query("https://b.r2.dev/c/v.mp4?X-Amz-Signature=abc"), "https://b.r2.dev/c/v.mp4")

    def test_extraer_videos_savevideo_y_vhs(self):
        self.assertEqual(len(h.extraer_videos(historial_ok("images"))), 1)
        self.assertEqual(len(h.extraer_videos(historial_ok("gifs"))), 1)
        sin_video = {"outputs": {"1": {"images": [{"filename": "a.png", "type": "output"}]}}}
        self.assertEqual(h.extraer_videos(sin_video), [])

    def test_error_oom(self):
        estado = {"status_str": "error", "messages": [["execution_error", {
            "node_id": "16", "node_type": "SamplerCustomAdvanced",
            "exception_type": "torch.OutOfMemoryError",
            "exception_message": "CUDA out of memory. Tried to allocate 2.00 GiB"}]]}
        self.assertIn("sin memoria", h.describir_error_ejecucion(estado))

    def test_nombre_seguro(self):
        self.assertEqual(h.nombre_seguro("../../etc/pass wd", None, "x.png"), "pass_wd")
        self.assertEqual(h.nombre_seguro(None, "audio/wav", "voz"), "voz.wav")
        self.assertEqual(h.nombre_seguro("hoja", "image/jpeg", "x"), "hoja.jpg")


# ---------------------------------------------------------------------------
class PruebasValidacion(unittest.TestCase):
    def base(self, **extra):
        e = {"flujo": "wan22-i2v", "prompt": "Una toma", "duracion_seg": 5, "aspecto": "9:16",
             "ancho": 720, "alto": 1280, "imagen_inicial": imagen(), "generar_audio": False}
        e.update(extra)
        return e

    def test_flujo_desconocido(self):
        with self.assertRaisesRegex(h.ErrorTrabajo, "Flujo desconocido"):
            h.validar_entrada(self.base(flujo="sora"))

    def test_falta_imagen(self):
        e = self.base()
        del e["imagen_inicial"]
        with self.assertRaisesRegex(h.ErrorTrabajo, "imagen_inicial"):
            h.validar_entrada(e)

    def test_h3_sin_referencias_usa_imagen(self):
        d = h.validar_entrada(self.base(flujo="h3-ref", duracion_seg=6, ancho=768, alto=1344))
        self.assertEqual(len(d["referencias"]), 1)
        self.assertEqual(d["fps"], 24)

    def test_h3_sin_nada_falla(self):
        e = self.base(flujo="h3-ref")
        del e["imagen_inicial"]
        with self.assertRaisesRegex(h.ErrorTrabajo, "referencias"):
            h.validar_entrada(e)

    def test_h3_maximo_9(self):
        with self.assertRaisesRegex(h.ErrorTrabajo, "máximo 9"):
            h.validar_entrada(self.base(flujo="h3-ref", referencias=[imagen()] * 10))

    def test_h3_fps_fijo(self):
        d = h.validar_entrada(self.base(flujo="h3-ref", fps=30))
        self.assertEqual(d["fps"], 24)

    def test_duracion_se_acota(self):
        self.assertEqual(h.validar_entrada(self.base(duracion_seg=30))["duracion_seg"], 10)
        self.assertEqual(h.validar_entrada(self.base(flujo="h3-ref", duracion_seg=1))["duracion_seg"], 4)

    def test_tipos_invalidos(self):
        with self.assertRaises(h.ErrorTrabajo):
            h.validar_entrada(self.base(duracion_seg="5"))
        with self.assertRaises(h.ErrorTrabajo):
            h.validar_entrada(self.base(generar_audio="si"))
        with self.assertRaises(h.ErrorTrabajo):
            h.validar_entrada(self.base(imagen_inicial={"nombre": "x"}))
        with self.assertRaises(h.ErrorTrabajo):
            h.validar_entrada(self.base(salida={"url_publica": "x"}))
        with self.assertRaises(h.ErrorTrabajo):
            h.validar_entrada(self.base(prompt="   "))

    def test_wan_no_genera_audio(self):
        self.assertFalse(h.validar_entrada(self.base(generar_audio=True))["generar_audio"])

    def test_imagen_no_pide_duracion_fps_ni_audio(self):
        d = h.validar_entrada({"flujo": "zimage-t2i", "prompt": "Una calle de noche"})
        self.assertEqual(d["tipo"], "imagen")
        self.assertEqual((d["ancho"], d["alto"]), (768, 1344))  # 9:16 por defecto
        self.assertIsNone(d["duracion_seg"])
        self.assertIsNone(d["fps"])
        self.assertFalse(d["generar_audio"])
        # si llegan campos de video (incluso con tipos raros) se ignoran
        d = h.validar_entrada({"flujo": "zimage-t2i", "prompt": "x", "duracion_seg": "5", "fps": 999,
                               "generar_audio": "si", "imagen_inicial": imagen(),
                               "audio_referencia": {"nombre": "v.wav", "mime": "audio/wav", "base64": WAV}})
        self.assertIsNone(d["imagen_inicial"])
        self.assertIsNone(d["audio_referencia"])
        self.assertEqual(d["referencias"], [])

    def test_imagen_respeta_tamano_y_semilla(self):
        d = h.validar_entrada({"flujo": "zimage-t2i", "prompt": "x", "ancho": 1080, "alto": 1920,
                               "semilla": 7, "prompt_negativo": "borroso"})
        self.assertEqual((d["ancho"], d["alto"], d["semilla"]), (1072, 1920, 7))  # múltiplo de 16; empate baja
        self.assertEqual(d["prompt_negativo"], "borroso")
        self.assertEqual(h.validar_entrada({"flujo": "qwen-edit-ref", "prompt": "x", "ancho": 720, "alto": 1280,
                                            "referencias": [imagen()]})["ancho"], 720)

    def test_qwen_referencias(self):
        e = {"flujo": "qwen-edit-ref", "prompt": "Picture 1 en el café de Picture 2"}
        with self.assertRaisesRegex(h.ErrorTrabajo, "referencias"):
            h.validar_entrada(e)
        with self.assertRaisesRegex(h.ErrorTrabajo, "máximo 3"):
            h.validar_entrada(dict(e, referencias=[imagen()] * 4))
        d = h.validar_entrada(dict(e, imagen_inicial=imagen("sola.png")))
        self.assertEqual([r["nombre"] for r in d["referencias"]], ["sola.png"])
        self.assertIsNone(d["imagen_inicial"])  # pasó a ser la referencia 1
        d = h.validar_entrada(dict(e, referencias=[imagen("a.png"), imagen("b.png")], imagen_inicial=imagen("c.png")))
        self.assertEqual([r["nombre"] for r in d["referencias"]], ["a.png", "b.png"])
        self.assertIsNone(d["imagen_inicial"])


# ---------------------------------------------------------------------------
class PruebasFlujos(unittest.TestCase):
    VALORES = {"PROMPT": "p", "PROMPT_NEGATIVO": "n", "ANCHO": 704, "ALTO": 1280, "ANCHO_MEDIO": 352,
               "ALTO_MEDIO": 640, "FRAMES": 121, "FPS": 24, "DURACION_SEG": 5.0417, "SEMILLA": 1,
               "IMAGEN_INICIAL": "img.png", "AUDIO_REFERENCIA": "voz.wav", "PREFIJO_SALIDA": "seriesia/t",
               "REFERENCIA_1": "r1.png", "REFERENCIA_2": "r2.png", "REFERENCIA_3": "", "FUERZA": 0.33}

    def revisar_grafo(self, prompt, es_imagen=False):
        texto = json.dumps(prompt, ensure_ascii=False)
        self.assertNotIn("{{", texto)
        self.assertNotIn('"_si"', texto)
        self.assertNotIn("_TODO", texto)
        for nid, nodo in prompt.items():
            self.assertEqual(set(nodo) - {"class_type", "inputs", "_meta"}, set())
            for k, v in nodo["inputs"].items():
                if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str):
                    self.assertIn(v[0], prompt, f"{nid}.{k} apunta a un nodo inexistente")
        salida = "SaveImage" if es_imagen else "SaveVideo"
        self.assertTrue(any(n["class_type"] == salida for n in prompt.values()))

    def test_todos_los_flujos_tienen_meta(self):
        for nombre in FLUJOS:
            meta = cargar(nombre)["_meta"]
            for clave in ("fuente_verificada", "fecha_verificacion", "nodos_requeridos", "modelos",
                          "carpetas_de_modelos", "vram_aproximada", "_TODO"):
                self.assertIn(clave, meta, f"{nombre} sin {clave}")
            for m in meta["modelos"]:
                self.assertEqual(meta["carpetas_de_modelos"][m["archivo"]], m["carpeta"])

    def test_todas_las_combinaciones(self):
        for nombre in FLUJOS:
            for img in (True, False):
                for aud in (True, False):
                    for gen in (True, False):
                        con_refs = nombre in ("h3-ref", "qwen-edit-ref", "qwen-edit-rapido")
                        b = {"imagen_inicial": img, "audio_referencia": aud, "generar_audio": gen,
                             "referencias": con_refs}
                        refs = ["r1.png", "r2.png"] if con_refs else []
                        with self.subTest(flujo=nombre, imagen=img, audio=aud, generar_audio=gen):
                            self.revisar_grafo(h.armar_prompt_api(cargar(nombre), self.VALORES, b, refs),
                                               es_imagen=nombre in FLUJOS_IMAGEN)

    def test_flujos_de_imagen_meta(self):
        for nombre in FLUJOS_IMAGEN:
            meta = cargar(nombre)["_meta"]
            self.assertEqual(meta["tipo"], "imagen")
            self.assertTrue(h.es_flujo_de_imagen(nombre))
            self.assertTrue(all("http" in f for f in meta["fuente_verificada"]))
            clases = {n["class_type"] for k, n in cargar(nombre).items() if k != "_meta"}
            self.assertLessEqual(clases, set(meta["nodos_requeridos"]) | {"LoadImage"})
        self.assertFalse(h.es_flujo_de_imagen("wan22-i2v"))

    def test_zimage_grafo(self):
        p = h.armar_prompt_api(cargar("zimage-t2i"), self.VALORES, {}, [])
        self.assertEqual(p["3"]["inputs"]["type"], "lumina2")
        self.assertEqual(p["7"]["class_type"], "EmptySD3LatentImage")
        self.assertEqual((p["7"]["inputs"]["width"], p["7"]["inputs"]["height"]), (704, 1280))
        k = p["8"]["inputs"]
        self.assertEqual((k["steps"], k["cfg"], k["sampler_name"], k["scheduler"], k["seed"]),
                         (8, 1.0, "res_multistep", "simple", 1))
        self.assertEqual(k["negative"], ["5", 0])
        self.assertEqual(p["2"]["inputs"]["shift"], 3.0)
        self.assertEqual(p["10"]["inputs"]["filename_prefix"], "seriesia/t")
        with self.assertRaises(h.ErrorTrabajo):  # no acepta referencias
            h.armar_prompt_api(cargar("zimage-t2i"), self.VALORES, {}, ["x.png"])

    def test_qwen_referencias_van_a_positivo_y_negativo(self):
        p = h.armar_prompt_api(cargar("qwen-edit-ref"), self.VALORES, {"referencias": True},
                               ["a.png", "b.png", "c.png"])
        for nid in ("6", "7"):
            self.assertEqual(p[nid]["class_type"], "TextEncodeQwenImageEditPlus")
            self.assertEqual(p[nid]["inputs"]["image1"], ["ref_1", 0])
            self.assertEqual(p[nid]["inputs"]["image2"], ["ref_2", 0])
            self.assertEqual(p[nid]["inputs"]["image3"], ["ref_3", 0])
            self.assertNotIn("image0", p[nid]["inputs"])
            self.assertEqual(p[nid]["inputs"]["vae"], ["5", 0])
        self.assertEqual(p["ref_3"]["inputs"]["image"], "c.png")
        self.assertEqual(p["6"]["inputs"]["prompt"], "p")
        self.assertEqual(p["7"]["inputs"]["prompt"], "n")
        self.assertEqual(p["4"]["inputs"]["type"], "qwen_image")
        k = p["11"]["inputs"]
        self.assertEqual((k["steps"], k["cfg"], k["sampler_name"]), (40, 4.0, "euler"))
        self.assertEqual((k["positive"], k["negative"], k["model"]), (["8", 0], ["9", 0], ["3", 0]))
        # con una sola referencia solo se conecta image1
        p = h.armar_prompt_api(cargar("qwen-edit-ref"), self.VALORES, {"referencias": True}, ["a.png"])
        self.assertEqual(p["6"]["inputs"]["image1"], ["ref_1", 0])
        self.assertNotIn("image2", p["6"]["inputs"])
        self.assertNotIn("ref_2", p)
        with self.assertRaisesRegex(h.ErrorTrabajo, "máximo 3"):
            h.armar_prompt_api(cargar("qwen-edit-ref"), self.VALORES, {}, ["a", "b", "c", "d"])

    def test_qwen_rapido_lora_lightning_4_pasos(self):
        p = h.armar_prompt_api(cargar("qwen-edit-rapido"), self.VALORES, {"referencias": True},
                               ["a.png", "b.png"])
        lora = p["14"]
        self.assertEqual(lora["class_type"], "LoraLoaderModelOnly")
        self.assertEqual(lora["inputs"]["lora_name"], "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors")
        self.assertEqual(lora["inputs"]["strength_model"], 1.0)
        self.assertEqual(lora["inputs"]["model"], ["3", 0])  # después de CFGNorm, como la plantilla
        self.assertEqual(p["3"]["inputs"]["model"], ["2", 0])
        self.assertEqual(p["2"]["inputs"]["shift"], 3.1)
        k = p["11"]["inputs"]
        self.assertEqual((k["steps"], k["cfg"], k["sampler_name"], k["scheduler"], k["model"]),
                         (4, 1.0, "euler", "simple", ["14", 0]))
        for nid in ("6", "7"):
            self.assertEqual(p[nid]["inputs"]["image1"], ["ref_1", 0])
            self.assertEqual(p[nid]["inputs"]["image2"], ["ref_2", 0])
            self.assertNotIn("image3", p[nid]["inputs"])
        meta = cargar("qwen-edit-rapido")["_meta"]
        self.assertEqual(meta["carpetas_de_modelos"]["Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"],
                         "loras")
        # fuera de la LoRA y el KSampler, el grafo es el mismo que qwen-edit-ref
        base = h.armar_prompt_api(cargar("qwen-edit-ref"), self.VALORES, {"referencias": True}, ["a.png", "b.png"])
        for nid in base:
            if nid != "11":
                self.assertEqual(p[nid], base[nid], nid)
        with self.assertRaisesRegex(h.ErrorTrabajo, "máximo 3"):
            h.armar_prompt_api(cargar("qwen-edit-rapido"), self.VALORES, {}, ["a", "b", "c", "d"])

    def test_extraer_imagenes(self):
        self.assertEqual(len(h.extraer_imagenes(historial_imagen())), 1)
        self.assertEqual(h.extraer_imagenes(historial_ok()), [])  # un .mp4 no es imagen
        self.assertEqual(h.extraer_videos(historial_imagen()), [])  # un .png no es video
        mixto = {"outputs": {"1": {"images": [{"filename": "p.png", "type": "temp"}]},
                             "2": {"images": [{"filename": "s.png", "type": "output"}]}}}
        self.assertEqual([i["filename"] for i in h.extraer_imagenes(mixto)], ["s.png"])
        self.assertEqual(h.mime_de_imagen("a.webp"), "image/webp")
        self.assertEqual(h.mime_de_imagen("a.png"), "image/png")

    def test_h3_referencias_y_guia(self):
        b = {"imagen_inicial": False, "audio_referencia": True, "generar_audio": True, "referencias": True}
        p = h.armar_prompt_api(cargar("h3-ref"), self.VALORES, b, ["a.png", "b.png", "c.png"])
        self.assertEqual(p["8"]["inputs"]["ref_images.ref_image_0"], ["ref_1", 0])
        self.assertEqual(p["8"]["inputs"]["ref_images.ref_image_2"], ["ref_3", 0])
        self.assertEqual(p["ref_2"]["inputs"]["image"], "b.png")
        self.assertEqual(p["12"]["inputs"]["conditioning"], ["11", 0])
        self.assertNotIn("image", p["11"]["inputs"])  # sin imagen inicial: la guía solo lleva audio
        self.assertEqual(p["11"]["inputs"]["audio"], ["10", 0])
        self.assertNotIn("9", p)
        self.assertIsInstance(p["8"]["inputs"]["length"], int)

    def test_h3_sin_guia(self):
        b = {"imagen_inicial": False, "audio_referencia": False, "generar_audio": False, "referencias": True}
        p = h.armar_prompt_api(cargar("h3-ref"), self.VALORES, b, ["a.png"])
        self.assertNotIn("11", p)
        self.assertEqual(p["12"]["inputs"]["conditioning"], ["8", 0])
        self.assertNotIn("18", p)
        self.assertNotIn("audio", p["19"]["inputs"])

    def test_ltx_audio_referencia(self):
        b = {"imagen_inicial": True, "audio_referencia": True, "generar_audio": True, "referencias": False}
        p = h.armar_prompt_api(cargar("ltx25-i2v"), self.VALORES, b, [])
        self.assertNotIn("14", p)
        self.assertEqual(p["20"]["inputs"]["audio_latent"], ["19", 0])
        self.assertEqual(p["16"]["inputs"]["duration"], 5.0417)
        self.assertEqual(p["12"]["inputs"]["width"], 352)
        b["audio_referencia"] = False
        p = h.armar_prompt_api(cargar("ltx25-i2v"), self.VALORES, b, [])
        self.assertEqual(p["20"]["inputs"]["audio_latent"], ["14", 0])
        for nid in ("15", "16", "17", "18", "19"):
            self.assertNotIn(nid, p)

    def test_wan5b_sin_imagen_es_texto_a_video(self):
        b = {"imagen_inicial": False, "audio_referencia": False, "generar_audio": False, "referencias": False}
        p = h.armar_prompt_api(cargar("wan22-ti2v-5b"), self.VALORES, b, [])
        self.assertNotIn("7", p)
        self.assertNotIn("start_image", p["8"]["inputs"])

    def test_referencias_en_flujo_que_no_las_acepta(self):
        with self.assertRaises(h.ErrorTrabajo):
            h.armar_prompt_api(cargar("wan22-i2v"), self.VALORES, {"imagen_inicial": True}, ["x.png"])

    def test_faltantes(self):
        b = {"imagen_inicial": True, "audio_referencia": False, "generar_audio": False}
        p = h.armar_prompt_api(cargar("wan22-i2v"), self.VALORES, b, [])
        info = object_info_completo()
        self.assertEqual(h.clases_faltantes(p, info), [])
        self.assertEqual(h.modelos_faltantes(p, info), [])
        del info["WanImageToVideo"]
        self.assertEqual(h.clases_faltantes(p, info), ["WanImageToVideo"])
        info["UNETLoader"]["input"]["required"]["unet_name"][1]["options"].remove(
            "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors")
        carpetas = cargar("wan22-i2v")["_meta"]["carpetas_de_modelos"]
        self.assertEqual(h.modelos_faltantes(p, info, carpetas),
                         ["wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors (carpeta models/diffusion_models/)"])


# ---------------------------------------------------------------------------
class PruebasHandlerCompleto(BaseConCarpetas):
    def correr(self, entrada, falso):
        os.makedirs(os.path.join(self.tmp, "output", "seriesia"), exist_ok=True)
        with open(os.path.join(self.tmp, "output", "seriesia", "x_00001.mp4"), "wb") as f:
            f.write(MP4)
        with mock.patch.object(h, "http", falso):
            return h.handler({"id": "job-1", "input": entrada})

    def test_wan_base64(self):
        falso = ComfyFalso(historial=historial_ok())
        r = self.correr({"flujo": "wan22-i2v", "prompt": "Ella gira", "duracion_seg": 5, "aspecto": "9:16",
                         "ancho": 720, "alto": 1280, "imagen_inicial": imagen(), "generar_audio": False,
                         "semilla": 99}, falso)
        self.assertNotIn("error", r, r.get("error"))
        self.assertEqual(base64.b64decode(r["video_base64"]), MP4)
        self.assertEqual((r["mime"], r["flujo"], r["ancho"], r["alto"], r["fps"]), ("video/mp4", "wan22-i2v", 720, 1280, 16))
        self.assertAlmostEqual(r["duracion_seg"], 81 / 16, places=3)
        self.assertIsInstance(r["segundos_gpu"], float)
        p = falso.prompt_enviado
        self.assertEqual(p["12"]["inputs"]["length"], 81)
        self.assertEqual(p["13"]["inputs"]["noise_seed"], 99)
        self.assertTrue(p["11"]["inputs"]["image"].startswith("job-1_"))
        # los archivos temporales se borran
        self.assertEqual(os.listdir(os.path.join(self.tmp, "input")), [])
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "output", "seriesia", "x_00001.mp4")))

    def test_h3_con_subida_put(self):
        falso = ComfyFalso(historial=historial_ok())
        r = self.correr({"flujo": "h3-ref", "prompt": "<Subject 1> sonríe.", "duracion_seg": 6, "aspecto": "9:16",
                         "ancho": 768, "alto": 1344, "referencias": [imagen("hoja.png"), imagen("hoja2.png")],
                         "audio_referencia": {"nombre": "voz.wav", "mime": "audio/wav", "base64": WAV},
                         "generar_audio": True,
                         "salida": {"url_subida": "https://bucket.ejemplo.com/v.mp4?firma=1"}}, falso)
        self.assertNotIn("error", r, r.get("error"))
        self.assertEqual(r["video_url"], "https://bucket.ejemplo.com/v.mp4")
        self.assertNotIn("video_base64", r)
        self.assertEqual(falso.puts[0][1], MP4)
        self.assertEqual(falso.puts[0][2]["Content-Type"], "video/mp4")
        p = falso.prompt_enviado
        self.assertEqual(p["8"]["inputs"]["length"], 158)  # 6 s -> 144 -> 158 (17k+5)
        self.assertIn("<Picture 2>", p["8"]["inputs"]["prompt"])
        self.assertTrue(p["10"]["inputs"]["audio"].endswith(".wav"))
        self.assertEqual(r["fps"], 24)

    def test_url_publica_y_descarga_por_url(self):
        falso = ComfyFalso(historial=historial_ok())
        r = self.correr({"flujo": "ltx25-i2v", "prompt": "x", "duracion_seg": 4, "ancho": 720, "alto": 1280,
                         "imagen_inicial": {"nombre": "f.png", "mime": "image/png", "url": "https://ejemplo.com/f.png"},
                         "generar_audio": True,
                         "salida": {"url_subida": "https://b.com/v.mp4?s=1", "url_publica": "https://cdn.com/v.mp4"}}, falso)
        self.assertEqual(r.get("video_url"), "https://cdn.com/v.mp4", r)
        self.assertEqual((r["ancho"], r["alto"]), (704, 1280))
        self.assertEqual(falso.prompt_enviado["12"]["inputs"]["width"], 352)
        self.assertEqual(falso.prompt_enviado["12"]["inputs"]["length"], 97)

    def test_video_por_view_si_no_esta_en_disco(self):
        falso = ComfyFalso(historial=historial_ok(archivo="seriesia/otro.mp4"))
        r = self.correr({"flujo": "wan22-ti2v-5b", "prompt": "paisaje", "duracion_seg": 3}, falso)
        self.assertEqual(base64.b64decode(r["video_base64"]), MP4)
        self.assertEqual(r["fps"], 24)

    def test_nodo_faltante(self):
        info = object_info_completo()
        del info["MiniMaxH3ReferenceToVideo"]
        r = self.correr({"flujo": "h3-ref", "prompt": "x", "duracion_seg": 5, "referencias": [imagen()]},
                        ComfyFalso(object_info=info))
        self.assertIn("Faltan nodos", r["error"])
        self.assertIn("MiniMaxH3ReferenceToVideo", r["error"])
        self.assertEqual(os.listdir(os.path.join(self.tmp, "input")), [])

    def test_modelo_faltante(self):
        info = object_info_completo()
        info["CLIPLoader"]["input"]["required"]["clip_name"][0].remove("qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors")
        r = self.correr({"flujo": "h3-ref", "prompt": "x", "duracion_seg": 5, "referencias": [imagen()]},
                        ComfyFalso(object_info=info))
        self.assertIn("Faltan modelos", r["error"])
        self.assertIn("qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors (carpeta models/text_encoders/)", r["error"])

    def test_oom(self):
        hist = {"status": {"status_str": "error", "completed": False, "messages": [["execution_error", {
            "node_id": "14", "node_type": "KSamplerAdvanced", "exception_type": "torch.OutOfMemoryError",
            "exception_message": "CUDA out of memory."}]]}, "outputs": {}}
        r = self.correr({"flujo": "wan22-i2v", "prompt": "x", "duracion_seg": 5, "imagen_inicial": imagen()},
                        ComfyFalso(historial=hist))
        self.assertIn("sin memoria", r["error"])

    def test_rechazo_de_validacion(self):
        cuerpo = {"error": {"type": "prompt_outputs_failed_validation", "message": "Prompt outputs failed validation",
                            "details": ""},
                  "node_errors": {"12": {"class_type": "WanImageToVideo",
                                         "errors": [{"message": "Value not in list", "details": "x"}]}}}
        r = self.correr({"flujo": "wan22-i2v", "prompt": "x", "duracion_seg": 5, "imagen_inicial": imagen()},
                        ComfyFalso(estado_prompt=400, cuerpo_prompt=cuerpo))
        self.assertIn("ComfyUI rechazó el flujo", r["error"])
        self.assertIn("WanImageToVideo", r["error"])

    def test_tiempo_agotado(self):
        falso = ComfyFalso(nunca_termina=True)
        with mock.patch.object(h, "TIEMPO_MAX_SEG", 0):
            r = self.correr({"flujo": "wan22-i2v", "prompt": "x", "duracion_seg": 5, "imagen_inicial": imagen()}, falso)
        self.assertIn("Se agotó el tiempo", r["error"])
        self.assertTrue(falso.interrumpido)

    def test_put_falla(self):
        r = self.correr({"flujo": "wan22-i2v", "prompt": "x", "duracion_seg": 5, "imagen_inicial": imagen(),
                         "salida": {"url_subida": "https://b.com/v.mp4?s=1"}},
                        ComfyFalso(historial=historial_ok(), estado_put=403))
        self.assertIn("HTTP 403", r["error"])

    def correr_imagen(self, entrada, falso, archivo="x_00001_.png"):
        os.makedirs(os.path.join(self.tmp, "output", "seriesia"), exist_ok=True)
        with open(os.path.join(self.tmp, "output", "seriesia", archivo), "wb") as f:
            f.write(base64.b64decode(PNG_1PX))
        with mock.patch.object(h, "http", falso):
            return h.handler({"id": "job-1", "input": entrada})

    def test_zimage_base64(self):
        falso = ComfyFalso(historial=historial_imagen())
        r = self.correr_imagen({"flujo": "zimage-t2i", "prompt": "Retrato de una mujer en la lluvia",
                                "semilla": 42, "aspecto": "9:16"}, falso)
        self.assertNotIn("error", r, r.get("error"))
        self.assertEqual(base64.b64decode(r["imagen_base64"]), base64.b64decode(PNG_1PX))
        # ancho/alto de la respuesta = los del PNG real (aquí 1×1), no los pedidos
        self.assertEqual((r["mime"], r["flujo"], r["ancho"], r["alto"], r["semilla"]),
                         ("image/png", "zimage-t2i", 1, 1, 42))
        self.assertIsInstance(r["segundos_gpu"], float)
        for clave in ("video_base64", "video_url", "duracion_seg", "fps", "frames"):
            self.assertNotIn(clave, r)
        p = falso.prompt_enviado
        self.assertEqual(p["8"]["inputs"]["seed"], 42)
        self.assertEqual(p["7"]["inputs"]["height"], 1344)
        self.assertEqual(p["4"]["inputs"]["text"], "Retrato de una mujer en la lluvia")
        self.assertEqual(os.listdir(os.path.join(self.tmp, "input")) if os.path.isdir(
            os.path.join(self.tmp, "input")) else [], [])
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "output", "seriesia", "x_00001_.png")))

    def test_qwen_con_subida_put_y_referencia_por_url(self):
        falso = ComfyFalso(historial=historial_imagen())
        r = self.correr_imagen({"flujo": "qwen-edit-ref", "prompt": "La mujer de Picture 1 en el bar de Picture 2",
                                "ancho": 768, "alto": 1344, "semilla": 5,
                                "referencias": [imagen("personaje.png"),
                                                {"nombre": "bar.png", "mime": "image/png",
                                                 "url": "https://ejemplo.com/bar.png"}],
                                "salida": {"url_subida": "https://bucket.ejemplo.com/i.png?firma=1"}}, falso)
        self.assertNotIn("error", r, r.get("error"))
        self.assertEqual(r["imagen_url"], "https://bucket.ejemplo.com/i.png")
        self.assertNotIn("imagen_base64", r)
        self.assertEqual(falso.puts[0][2]["Content-Type"], "image/png")
        p = falso.prompt_enviado
        self.assertTrue(p["ref_1"]["inputs"]["image"].endswith("personaje.png"))
        self.assertTrue(p["ref_2"]["inputs"]["image"].endswith("bar.png"))
        self.assertEqual(p["7"]["inputs"]["image2"], ["ref_2", 0])
        self.assertEqual(p["7"]["inputs"]["prompt"], "")  # negativo por defecto de la plantilla
        self.assertEqual(p["11"]["inputs"]["seed"], 5)
        self.assertEqual(os.listdir(os.path.join(self.tmp, "input")), [])

    def test_qwen_negativo_e_imagen_inicial_como_referencia(self):
        falso = ComfyFalso(historial=historial_imagen(archivo="otra.png"))  # no está en disco: va por /view
        r = self.correr_imagen({"flujo": "qwen-edit-ref", "prompt": "x", "prompt_negativo": "borroso",
                                "imagen_inicial": imagen("hoja.png")}, falso)
        self.assertNotIn("error", r, r.get("error"))
        self.assertEqual(base64.b64decode(r["imagen_base64"]), base64.b64decode(PNG_1PX))
        p = falso.prompt_enviado
        self.assertEqual(p["7"]["inputs"]["prompt"], "borroso")
        self.assertTrue(p["ref_1"]["inputs"]["image"].endswith("hoja.png"))
        self.assertNotIn("ref_2", p)

    def test_qwen_rapido_probador(self):
        falso = ComfyFalso(historial=historial_imagen())
        r = self.correr_imagen({"flujo": "qwen-edit-rapido", "prompt": "Dress the woman from Picture 1 in the corset from Picture 2",
                                "prompt_negativo": "deformed", "ancho": 768, "alto": 1024, "semilla": 11,
                                "referencias": [imagen("clienta.png"),
                                                {"nombre": "corse.jpg", "mime": "image/jpeg",
                                                 "url": "https://ejemplo.com/corse.jpg"}]}, falso)
        self.assertNotIn("error", r, r.get("error"))
        self.assertEqual((r["mime"], r["flujo"], r["ancho"], r["alto"], r["semilla"]),
                         ("image/png", "qwen-edit-rapido", 1, 1, 11))
        self.assertEqual(base64.b64decode(r["imagen_base64"]), base64.b64decode(PNG_1PX))
        p = falso.prompt_enviado
        self.assertTrue(p["ref_1"]["inputs"]["image"].endswith("clienta.png"))
        self.assertTrue(p["ref_2"]["inputs"]["image"].endswith("corse.jpg"))
        self.assertEqual((p["11"]["inputs"]["steps"], p["11"]["inputs"]["seed"]), (4, 11))
        self.assertEqual(p["7"]["inputs"]["prompt"], "deformed")
        self.assertEqual(os.listdir(os.path.join(self.tmp, "input")), [])

    def test_qwen_rapido_sin_lora_en_el_volumen(self):
        info = object_info_completo()
        info["LoraLoaderModelOnly"]["input"]["required"]["lora_name"][0].remove(
            "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors")
        r = self.correr_imagen({"flujo": "qwen-edit-rapido", "prompt": "x", "referencias": [imagen()]},
                               ComfyFalso(object_info=info))
        self.assertIn("Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors (carpeta models/loras/)", r["error"])

    def test_imagen_sin_salida(self):
        r = self.correr_imagen({"flujo": "zimage-t2i", "prompt": "x"}, ComfyFalso(historial=historial_ok()))
        self.assertIn("ninguna imagen", r["error"])

    def test_imagen_nodo_faltante(self):
        info = object_info_completo()
        del info["TextEncodeQwenImageEditPlus"]
        r = self.correr_imagen({"flujo": "qwen-edit-ref", "prompt": "x", "referencias": [imagen()]},
                               ComfyFalso(object_info=info))
        self.assertIn("TextEncodeQwenImageEditPlus", r["error"])
        self.assertEqual(os.listdir(os.path.join(self.tmp, "input")), [])

    def test_imagen_modelo_faltante(self):
        info = object_info_completo()
        info["CLIPLoader"]["input"]["required"]["clip_name"][0].remove("qwen_3_4b.safetensors")
        r = self.correr_imagen({"flujo": "zimage-t2i", "prompt": "x"}, ComfyFalso(object_info=info))
        self.assertIn("qwen_3_4b.safetensors (carpeta models/text_encoders/)", r["error"])

    def test_entrada_vacia(self):
        self.assertIn("error", h.handler({"id": "x"}))
        self.assertIn("error", h.handler(None))

    def test_entrada_ejemplo_es_valida(self):
        for archivo in ("entrada-ejemplo.json", "entrada-ejemplo-imagen.json"):
            with open(os.path.join(RAIZ, "pruebas", archivo), encoding="utf-8") as f:
                ejemplo = json.load(f)
            d = h.validar_entrada(ejemplo["input"])
            self.assertEqual(d["flujo"], ejemplo["input"]["flujo"])


if __name__ == "__main__":
    unittest.main()
