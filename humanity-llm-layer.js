/* ============================================================
 * HUMANITY LLM LAYER (Fase 2, opcional)
 * ------------------------------------------------------------
 * Capa de lenguaje chico corriendo 100% en el navegador via
 * Transformers.js (WebGPU con fallback a WASM). No reemplaza al
 * motor de humanity-engine.js: el retrieval de Fase 1 sigue siendo
 * la unica fuente de verdad factual. Esta capa solo reformula, con
 * mas soltura, un fragmento que el retrieval ya trajo — nunca se la
 * llama con una pregunta libre sin ese fragmento.
 *
 * Se carga de forma perezosa: la libreria Transformers.js recien se
 * baja (via import() dinamico desde CDN) cuando alguien pide
 * "activar modulo" en modo HUMANITY. Si algo falla en cualquier
 * punto (sin WebGPU, sin memoria, error de red, modelo corrupto en
 * cache, generacion demasiado lenta), esta capa se marca como no
 * disponible y HUMANITY sigue funcionando en Fase 1 sin romperse.
 *
 * No usa ninguna API externa paga: solo la libreria (CDN, JS puro)
 * y los pesos del modelo (Hugging Face Hub), cacheados por el
 * navegador via la Cache API que ya maneja Transformers.js.
 * ============================================================ */

const HumanityLLM = (function () {
  'use strict';

  /* ---------------- configuracion ajustable ---------------- */
  const CONFIG = {
    // CDN con la libreria empaquetada como modulo ES, sin necesidad de bundler
    transformersUrl: 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0',
    // modelo chico multilingue, cuantizado a 4 bits para el grueso de los pesos
    modelId: 'onnx-community/Qwen2.5-0.5B-Instruct',
    dtype: 'q4',
    approxSizeLabel: '~400 MB (aprox., varia segun cuantizacion de embeddings)',
    maxNewTokens: 120,
    generationTimeoutMs: 20000, // si tarda mas que esto, se degrada a Fase 1 para esa respuesta puntual
    systemPromptTemplate:
      'Sos HUMANITY, un archivo de respaldo daniado que sobrevivio a la desaparicion de la humanidad. ' +
      'Respondes UNICAMENTE en base al siguiente fragmento recuperado del archivo, sin inventar datos ' +
      'que no esten en el. Si el fragmento no alcanza para responder del todo, decilo asi. Tono contenido, ' +
      'de archivo post-colapso, breve (2 a 4 oraciones), en español.\n\nFRAGMENTO RECUPERADO:\n"""\n{fragmento}\n"""',
  };

  /* estado interno: nunca se expone directo, solo via las funciones de abajo */
  const state = {
    status: 'idle', // idle | loading | ready | unavailable
    generator: null,
    backend: null, // 'webgpu' | 'wasm'
    lastError: null,
  };

  function detectCandidateBackends() {
    const hasWebGPU = typeof navigator !== 'undefined' && !!navigator.gpu;
    return hasWebGPU ? ['webgpu', 'wasm'] : ['wasm'];
  }

  function isAvailable() {
    return state.status === 'ready';
  }

  function getStatus() {
    return state.status;
  }

  /* Dispara la carga del modelo. `onProgress(pct, fileName)` se llama en
   * cada evento de progreso que reporta Transformers.js mientras baja
   * los archivos del modelo (o instantaneo si ya estan en cache).
   * Prueba los backends disponibles en orden (WebGPU primero si hay
   * soporte) y si todos fallan, deja la capa en 'unavailable' sin tirar
   * la excepcion hacia quien llama. */
  async function activate(onProgress) {
    if (state.status === 'ready') return { ok: true, cached: true, backend: state.backend };
    if (state.status === 'loading') return { ok: false, reason: 'ya-cargando' };

    state.status = 'loading';

    let transformersLib;
    try {
      transformersLib = await import(CONFIG.transformersUrl);
    } catch (err) {
      console.warn('[HUMANITY-LLM] no se pudo cargar la libreria Transformers.js:', err);
      state.status = 'unavailable';
      state.lastError = err;
      return { ok: false, reason: 'error-libreria', error: err };
    }

    const { pipeline, env } = transformersLib;
    env.allowLocalModels = false; // siempre HF Hub + cache del navegador, nunca un backend propio

    const candidates = detectCandidateBackends();
    for (const backend of candidates) {
      try {
        const generator = await pipeline('text-generation', CONFIG.modelId, {
          dtype: CONFIG.dtype,
          device: backend,
          progress_callback: (data) => {
            if (onProgress && data.status === 'progress' && typeof data.progress === 'number') {
              onProgress(Math.round(data.progress), data.file || '');
            }
          },
        });
        state.generator = generator;
        state.backend = backend;
        state.status = 'ready';
        return { ok: true, cached: false, backend };
      } catch (err) {
        console.warn('[HUMANITY-LLM] fallo backend "' + backend + '":', err);
        state.lastError = err;
      }
    }

    state.status = 'unavailable';
    state.generator = null;
    return { ok: false, reason: 'error-carga', error: state.lastError };
  }

  function withTimeout(promise, ms) {
    return Promise.race([
      promise,
      new Promise((_, reject) => setTimeout(() => reject(new Error('timeout de generacion')), ms)),
    ]);
  }

  /* El formato de salida de "text-generation" con chat template puede ser
   * [{ generated_text: "..." }] o [{ generated_text: [...mensajes...] }]
   * segun el modelo/version. Se contemplan ambos casos. */
  function extractGeneratedText(output) {
    if (!Array.isArray(output) || output.length === 0) return null;
    const generated = output[0].generated_text;
    if (typeof generated === 'string') return generated;
    if (Array.isArray(generated)) {
      const last = generated[generated.length - 1];
      return last && typeof last.content === 'string' ? last.content : null;
    }
    return null;
  }

  /* UNICA forma de usar el modelo: siempre requiere un fragmento ya
   * encontrado por retrieval (humanity-engine.js es responsable de nunca
   * llamar esto sin un match real). Devuelve null ante cualquier falla
   * (no disponible, error de generacion, timeout) para que quien llama
   * pueda degradar a la respuesta de Fase 1 sin romper nada. */
  async function rephrase(fragment, questionHint) {
    if (!isAvailable() || !fragment) return null;

    const systemPrompt = CONFIG.systemPromptTemplate.replace('{fragmento}', fragment);
    const messages = [
      { role: 'system', content: systemPrompt },
      { role: 'user', content: questionHint || 'Contame que dice el fragmento.' },
    ];

    try {
      const output = await withTimeout(
        state.generator(messages, {
          max_new_tokens: CONFIG.maxNewTokens,
          temperature: 0.7,
          do_sample: true,
          return_full_text: false,
        }),
        CONFIG.generationTimeoutMs
      );
      const text = extractGeneratedText(output);
      return text ? text.trim() : null;
    } catch (err) {
      console.warn('[HUMANITY-LLM] fallo la generacion, se degrada a Fase 1:', err);
      return null;
    }
  }

  /* API publica del modulo */
  return {
    activate,
    rephrase,
    isAvailable,
    getStatus,
    CONFIG,
  };
})();

window.HumanityLLM = HumanityLLM;
