"""Pruebas de los flujos de la máquina de catálogos (sin GPU ni ComfyUI).

catalogo-prenda, catalogo-modelo, catalogo-angulos y mejorar-2k: reglas, texto base,
grafo armado (calidad rápida/alta, LoRA de ángulos, referencias opcionales) y el
handler completo con ComfyUI simulado.

Correr:  python -m pytest -q E:/series-ia/gpu/pruebas
"""

import base64
import json
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_handler import (  # noqa: E402
    FLUJOS_CATALOGO, PNG_1PX, BaseConCarpetas, ComfyFalso, cargar, h, historial_imagen, imagen,
    object_info_completo,
)

VALORES = {"PROMPT": "p", "PROMPT_NEGATIVO": "", "ANCHO": 864, "ALTO": 1536, "SEMILLA": 7,
           "PREFIJO_SALIDA": "seriesia/t", "IMAGEN_INICIAL": "", "AUDIO_REFERENCIA": "",
           "ANCHO_MEDIO": 432, "ALTO_MEDIO": 768, "REFERENCIA_1": "uno.png", "REFERENCIA_2": "dos.png",
           "FUERZA": 0.25}


def ruta_panel(meta):
    """Archivo del flujo de panel de origen (solo existe en E:/series-ia/gpu, no en el repo de la imagen)."""
    return os.path.join(os.path.dirname(h.__file__), meta["origen_panel"][len("gpu/"):])


HAY_PANEL = os.path.isdir(os.path.join(os.path.dirname(h.__file__), "flujos-panel"))


def armar(flujo, banderas, refs=("uno.png", "dos.png")):
    return h.armar_prompt_api(cargar(flujo), VALORES, banderas, list(refs))


class PruebasReglasCatalogo(unittest.TestCase):
    def test_meta_coincide_con_las_reglas(self):
        for flujo in FLUJOS_CATALOGO:
            meta = cargar(flujo)["_meta"]
            reglas = h.FLUJOS[flujo]
            self.assertEqual(meta["referencias"]["modo"], "marcadores")
            self.assertEqual(meta["referencias"]["maximo"], reglas["max_referencias"], flujo)
            self.assertEqual(meta["referencias"]["minimo"], reglas["min_referencias"], flujo)
            self.assertTrue(reglas["prompt_opcional"])
            self.assertIn("origen_panel", meta)
            if HAY_PANEL:
                self.assertTrue(os.path.isfile(ruta_panel(meta)), meta["origen_panel"])
            # sin nodos que solo existen en la interfaz de ComfyUI
            clases = {n["class_type"] for k, n in cargar(flujo).items() if k != "_meta"}
            self.assertFalse(clases & {"MarkdownNote", "PrimitiveInt", "PrimitiveFloat", "Note"})

    def test_cada_angulo_tiene_su_texto(self):
        meta = cargar("catalogo-angulos")["_meta"]
        self.assertEqual(set(meta["prompts"]), set(h.FLUJOS["catalogo-angulos"]["angulos"]))
        self.assertLessEqual(set(meta["angulos_con_lora"]), set(meta["prompts"]))
        for a in meta["angulos_con_lora"]:
            self.assertTrue(meta["prompts"][a].startswith("<sks> "), a)  # frase de la LoRA Multiángulo
        for a in set(meta["prompts"]) - set(meta["angulos_con_lora"]):
            self.assertNotIn("<sks>", meta["prompts"][a])

    @unittest.skipUnless(HAY_PANEL, "sin gpu/flujos-panel (repo de la imagen)")
    def test_textos_iguales_al_panel(self):
        """Los textos base son los mismos del flujo de panel (no se reescribieron)."""
        for flujo, clave in (("catalogo-prenda", "base"), ("catalogo-modelo", "refs_2"), ("mejorar-2k", "base")):
            meta = cargar(flujo)["_meta"]
            with open(ruta_panel(meta), encoding="utf-8") as f:
                panel = json.load(f)
            textos = [w for n in panel["nodes"] if n["type"] in ("TextEncodeQwenImageEditPlus", "CLIPTextEncode")
                      for w in n.get("widgets_values") or [] if isinstance(w, str) and w]
            self.assertIn(meta["prompts"][clave], textos, flujo)
        meta = cargar("catalogo-angulos")["_meta"]
        with open(ruta_panel(meta), encoding="utf-8") as f:
            panel = json.load(f)
        textos = {w for n in panel["nodes"] if n["type"] == "TextEncodeQwenImageEditPlus"
                  for w in n.get("widgets_values") or [] if isinstance(w, str) and w}
        self.assertEqual(set(meta["prompts"].values()), textos)


class PruebasValidacionCatalogo(unittest.TestCase):
    def test_prompt_opcional_y_calidad_por_defecto(self):
        d = h.validar_entrada({"flujo": "catalogo-prenda", "referencias": [imagen("m.png"), imagen("p.png")]})
        self.assertEqual((d["prompt"], d["calidad"], d["ancho"], d["alto"]), ("", "rapida", 864, 1536))
        d = h.validar_entrada({"flujo": "catalogo-prenda", "calidad": "alta", "prompt": "  body negro  ",
                               "referencias": [imagen(), imagen()]})
        self.assertEqual((d["prompt"], d["calidad"]), ("body negro", "alta"))
        with self.assertRaisesRegex(h.ErrorTrabajo, "calidad"):
            h.validar_entrada({"flujo": "catalogo-prenda", "calidad": "ultra", "referencias": [imagen(), imagen()]})
        with self.assertRaisesRegex(h.ErrorTrabajo, "texto"):
            h.validar_entrada({"flujo": "catalogo-prenda", "prompt": 5, "referencias": [imagen(), imagen()]})

    def test_cantidad_de_referencias(self):
        with self.assertRaisesRegex(h.ErrorTrabajo, "necesita 2 imágenes"):
            h.validar_entrada({"flujo": "catalogo-prenda", "referencias": [imagen()]})
        with self.assertRaisesRegex(h.ErrorTrabajo, "máximo 2"):
            h.validar_entrada({"flujo": "catalogo-modelo", "referencias": [imagen()] * 3})
        with self.assertRaisesRegex(h.ErrorTrabajo, "al menos una imagen"):
            h.validar_entrada({"flujo": "mejorar-2k"})
        # la imagen_inicial hace de referencia 1
        d = h.validar_entrada({"flujo": "mejorar-2k", "imagen_inicial": imagen("f.png")})
        self.assertEqual(len(d["referencias"]), 1)
        self.assertIsNone(d["imagen_inicial"])

    def test_angulo_obligatorio_y_valido(self):
        with self.assertRaisesRegex(h.ErrorTrabajo, "angulo"):
            h.validar_entrada({"flujo": "catalogo-angulos", "referencias": [imagen()]})
        with self.assertRaisesRegex(h.ErrorTrabajo, "angulo"):
            h.validar_entrada({"flujo": "catalogo-angulos", "angulo": "arriba", "referencias": [imagen()]})
        d = h.validar_entrada({"flujo": "catalogo-angulos", "angulo": "espalda", "referencias": [imagen()]})
        self.assertEqual(d["angulo"], "espalda")
        # los demás flujos ignoran «angulo»
        d = h.validar_entrada({"flujo": "catalogo-prenda", "angulo": "espalda", "referencias": [imagen(), imagen()]})
        self.assertIsNone(d["angulo"])

    def test_fuerza_de_mejorar_2k(self):
        d = h.validar_entrada({"flujo": "mejorar-2k", "referencias": [imagen()]})
        self.assertEqual((d["fuerza"], d["calidad"]), (0.33, None))
        self.assertEqual(h.validar_entrada({"flujo": "mejorar-2k", "fuerza": 0.2, "referencias": [imagen()]})["fuerza"], 0.2)
        for malo in (0.05, 0.9, "mucho", True):
            with self.assertRaises(h.ErrorTrabajo):
                h.validar_entrada({"flujo": "mejorar-2k", "fuerza": malo, "referencias": [imagen()]})

    def test_prompt_base_vacio(self):
        with self.assertRaisesRegex(h.ErrorTrabajo, "prompt_base"):
            h.validar_entrada({"flujo": "mejorar-2k", "prompt_base": " ", "referencias": [imagen()]})

    def test_los_flujos_viejos_siguen_pidiendo_prompt(self):
        with self.assertRaisesRegex(h.ErrorTrabajo, "Falta el «prompt»"):
            h.validar_entrada({"flujo": "qwen-edit-rapido", "referencias": [imagen()]})


class PruebasTextoBase(unittest.TestCase):
    def test_base_mas_agregado(self):
        meta = cargar("catalogo-prenda")["_meta"]
        base = meta["prompts"]["base"]
        self.assertEqual(h.prompt_de_catalogo(meta, {"prompt": ""}, 2), base)
        self.assertEqual(h.prompt_de_catalogo(meta, {"prompt": "Black lace body."}, 2), base + " Black lace body.")
        self.assertEqual(h.prompt_de_catalogo(meta, {"prompt": "x", "prompt_base": "Otro texto."}, 2), "Otro texto. x")

    def test_cambiar_modelo_con_y_sin_foto(self):
        meta = cargar("catalogo-modelo")["_meta"]
        self.assertTrue(h.prompt_de_catalogo(meta, {"prompt": ""}, 2).startswith("Replace the woman in image 1 with the woman from image 2"))
        t = h.prompt_de_catalogo(meta, {"prompt": "a tall woman with short red hair"}, 1)
        self.assertIn("with a different woman: a tall woman with short red hair. Keep the outfit", t)
        self.assertNotIn("{EXTRA}", t)
        with self.assertRaisesRegex(h.ErrorTrabajo, "describirla"):
            h.prompt_de_catalogo(meta, {"prompt": " "}, 1)

    def test_texto_por_angulo(self):
        meta = cargar("catalogo-angulos")["_meta"]
        self.assertTrue(h.prompt_de_catalogo(meta, {"angulo": "espalda", "prompt": ""}, 1).startswith("<sks> back view"))
        self.assertTrue(h.prompt_de_catalogo(meta, {"angulo": "detalle", "prompt": ""}, 1).startswith("Extreme close-up"))

    def test_banderas(self):
        meta = cargar("catalogo-angulos")["_meta"]
        b = h.banderas_de_catalogo(meta, {"calidad": "rapida", "angulo": "lado"}, 1)
        self.assertEqual(b, {"referencia_1": True, "lightning": True, "lora_angulos": True})
        b = h.banderas_de_catalogo(meta, {"calidad": "alta", "angulo": "caminando"}, 1)
        self.assertEqual((b["lightning"], b["lora_angulos"]), (False, False))

    def test_dimensiones_png(self):
        self.assertEqual(h.dimensiones_png(base64.b64decode(PNG_1PX)), (1, 1))
        self.assertIsNone(h.dimensiones_png(b"no es png"))


class PruebasGrafosCatalogo(unittest.TestCase):
    def revisar(self, p):
        texto = json.dumps(p, ensure_ascii=False)
        self.assertNotIn("{{", texto)
        self.assertNotIn('"_si"', texto)
        for nid, nodo in p.items():
            for k, v in nodo["inputs"].items():
                if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str):
                    self.assertIn(v[0], p, f"{nid}.{k} apunta a un nodo inexistente")
        self.assertEqual(sum(n["class_type"] == "SaveImage" for n in p.values()), 1)

    def test_prenda_rapida_y_alta(self):
        p = armar("catalogo-prenda", {"lightning": True, "referencia_1": True, "referencia_2": True})
        self.revisar(p)
        k = p["11"]["inputs"]
        self.assertEqual((k["model"], k["steps"], k["cfg"], k["seed"]), (["14", 0], 4, 1.0, 7))
        self.assertEqual(p["14"]["inputs"]["model"], ["3", 0])
        self.assertEqual(p["20"]["inputs"]["image"], "uno.png")  # modelo
        self.assertEqual(p["21"]["inputs"]["image"], "dos.png")  # prenda
        self.assertEqual(p["22"]["inputs"], {"image": ["20", 0], "upscale_method": "lanczos", "width": 864,
                                             "height": 1536, "crop": "center"})
        for nid in ("6", "7"):
            self.assertEqual((p[nid]["inputs"]["image1"], p[nid]["inputs"]["image2"]), (["22", 0], ["21", 0]))
        self.assertNotIn("ref_1", p)  # modo marcadores: el handler no agrega LoadImage
        p = armar("catalogo-prenda", {"lightning": False})
        self.revisar(p)
        k = p["11"]["inputs"]
        self.assertEqual((k["model"], k["steps"], k["cfg"]), (["3", 0], 20, 4.0))
        self.assertNotIn("14", p)

    def test_modelo_sin_foto_2(self):
        p = armar("catalogo-modelo", {"lightning": True, "referencia_1": True}, refs=("uno.png",))
        self.revisar(p)
        self.assertNotIn("21", p)
        for nid in ("6", "7"):
            self.assertNotIn("image2", p[nid]["inputs"])
            self.assertEqual(p[nid]["inputs"]["image1"], ["22", 0])
        p = armar("catalogo-modelo", {"lightning": True, "referencia_1": True, "referencia_2": True})
        self.assertEqual(p["6"]["inputs"]["image2"], ["21", 0])

    def test_angulos_con_y_sin_lora(self):
        p = armar("catalogo-angulos", {"lightning": True, "lora_angulos": True}, refs=("uno.png",))
        self.revisar(p)
        self.assertEqual(p["15"]["inputs"]["lora_name"], "qwen-image-edit-2511-multiple-angles-lora.safetensors")
        self.assertEqual(p["15"]["inputs"]["model"], ["1", 0])
        self.assertEqual(p["2"]["inputs"]["model"], ["15", 0])
        self.assertEqual(p["11"]["inputs"]["model"], ["14", 0])
        p = armar("catalogo-angulos", {"lightning": False, "lora_angulos": False}, refs=("uno.png",))
        self.revisar(p)
        self.assertNotIn("15", p)
        self.assertNotIn("14", p)
        self.assertEqual(p["2"]["inputs"]["model"], ["1", 0])
        self.assertEqual(p["11"]["inputs"]["steps"], 20)

    def test_mejorar_2k(self):
        p = armar("mejorar-2k", {}, refs=("uno.png",))
        self.revisar(p)
        self.assertEqual(p["2"]["inputs"]["image"], "uno.png")
        self.assertEqual(p["3"]["inputs"]["resolution_steps"], 16)
        self.assertEqual(p["4"]["inputs"]["model_name"], "RealESRGAN_x4plus.safetensors")
        self.assertEqual(p["6"]["inputs"]["scale_by"], 0.5)
        k = p["14"]["inputs"]
        self.assertEqual((k["denoise"], k["steps"], k["cfg"], k["sampler_name"], k["scheduler"]),
                         (0.25, 5, 1.0, "dpmpp_2m_sde", "beta"))
        self.assertEqual(k["latent_image"], ["13", 0])
        self.assertEqual(p["8"]["inputs"]["type"], "lumina2")

    def test_mas_referencias_que_el_maximo(self):
        with self.assertRaisesRegex(h.ErrorTrabajo, "máximo 1"):
            armar("mejorar-2k", {}, refs=("a.png", "b.png"))

    def test_extension_de_modelo_alternativa(self):
        p = armar("mejorar-2k", {}, refs=("uno.png",))
        info = object_info_completo()
        opciones = info["UpscaleModelLoader"]["input"]["required"]["model_name"][0]
        opciones.remove("RealESRGAN_x4plus.safetensors")
        opciones.append("RealESRGAN_x4plus.pth")
        self.assertEqual(h.resolver_extensiones_de_modelo(p, info),
                         ["RealESRGAN_x4plus.safetensors → RealESRGAN_x4plus.pth"])
        self.assertEqual(p["4"]["inputs"]["model_name"], "RealESRGAN_x4plus.pth")
        self.assertEqual(h.modelos_faltantes(p, info), [])


class PruebasHandlerCatalogo(BaseConCarpetas):
    def correr(self, entrada, falso=None):
        falso = falso or ComfyFalso(historial=historial_imagen())
        os.makedirs(os.path.join(self.tmp, "output", "seriesia"), exist_ok=True)
        with open(os.path.join(self.tmp, "output", "seriesia", "x_00001_.png"), "wb") as f:
            f.write(base64.b64decode(PNG_1PX))
        with mock.patch.object(h, "http", falso):
            return h.handler({"id": "job-cat", "input": entrada}), falso

    def test_vestir_modelo(self):
        r, falso = self.correr({"flujo": "catalogo-prenda", "prompt": "Black lace body.", "semilla": 3,
                                "referencias": [imagen("modelo.png"), imagen("prenda.png")]})
        self.assertNotIn("error", r, r.get("error"))
        self.assertEqual((r["flujo"], r["calidad"], r["semilla"]), ("catalogo-prenda", "rapida", 3))
        self.assertEqual(base64.b64decode(r["imagen_base64"]), base64.b64decode(PNG_1PX))
        p = falso.prompt_enviado
        self.assertTrue(p["20"]["inputs"]["image"].endswith("modelo.png"))
        self.assertTrue(p["21"]["inputs"]["image"].endswith("prenda.png"))
        self.assertTrue(p["6"]["inputs"]["prompt"].startswith("Dress the woman from image 1"))
        self.assertTrue(p["6"]["inputs"]["prompt"].endswith("Black lace body."))
        self.assertEqual(p["11"]["inputs"]["steps"], 4)
        self.assertEqual(os.listdir(os.path.join(self.tmp, "input")), [])  # entradas borradas

    def test_cambiar_modelo_por_descripcion(self):
        r, falso = self.correr({"flujo": "catalogo-modelo", "calidad": "alta", "prompt": "a woman with curly black hair",
                                "referencias": [imagen("puesta.png")]})
        self.assertNotIn("error", r, r.get("error"))
        p = falso.prompt_enviado
        self.assertNotIn("21", p)
        self.assertIn("a woman with curly black hair", p["6"]["inputs"]["prompt"])
        self.assertEqual((p["11"]["inputs"]["steps"], p["11"]["inputs"]["cfg"]), (20, 4.0))

    def test_cambiar_modelo_sin_foto_ni_texto(self):
        r, _ = self.correr({"flujo": "catalogo-modelo", "referencias": [imagen()]})
        self.assertIn("describirla", r["error"])

    def test_un_angulo(self):
        r, falso = self.correr({"flujo": "catalogo-angulos", "angulo": "tres_cuartos", "referencias": [imagen()]})
        self.assertNotIn("error", r, r.get("error"))
        self.assertEqual(r["angulo"], "tres_cuartos")
        p = falso.prompt_enviado
        self.assertTrue(p["6"]["inputs"]["prompt"].startswith("<sks> front-left quarter view"))
        self.assertIn("15", p)

    def test_mejorar_2k_con_pth_y_tamano_real(self):
        info = object_info_completo()
        opciones = info["UpscaleModelLoader"]["input"]["required"]["model_name"][0]
        opciones[:] = ["RealESRGAN_x4plus.pth"]
        r, falso = self.correr({"flujo": "mejorar-2k", "fuerza": 0.2, "imagen_inicial": imagen("final.png")},
                               ComfyFalso(object_info=info, historial=historial_imagen()))
        self.assertNotIn("error", r, r.get("error"))
        self.assertEqual((r["ancho"], r["alto"], r["fuerza"]), (1, 1, 0.2))  # tamaño del PNG real
        p = falso.prompt_enviado
        self.assertEqual(p["4"]["inputs"]["model_name"], "RealESRGAN_x4plus.pth")
        self.assertEqual(p["14"]["inputs"]["denoise"], 0.2)
        self.assertEqual(p["11"]["inputs"]["text"], cargar("mejorar-2k")["_meta"]["prompts"]["base"])

    def test_falta_la_lora_de_angulos(self):
        info = object_info_completo()
        info["LoraLoaderModelOnly"]["input"]["required"]["lora_name"][0].remove(
            "qwen-image-edit-2511-multiple-angles-lora.safetensors")
        r, _ = self.correr({"flujo": "catalogo-angulos", "angulo": "frente", "referencias": [imagen()]},
                           ComfyFalso(object_info=info))
        self.assertIn("qwen-image-edit-2511-multiple-angles-lora.safetensors (carpeta models/loras/)", r["error"])
        # «detalle» no usa esa LoRA: funciona aunque falte
        r, _ = self.correr({"flujo": "catalogo-angulos", "angulo": "detalle", "referencias": [imagen()]},
                           ComfyFalso(object_info=info, historial=historial_imagen()))
        self.assertNotIn("error", r, r.get("error"))


if __name__ == "__main__":
    unittest.main()
