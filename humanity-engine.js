/* ============================================================
 * HUMANITY - motor de archivo/respaldo
 * ------------------------------------------------------------
 * Todo client-side, sin llamadas a APIs externas de IA.
 * Expone un unico objeto global: window.Humanity
 *
 * Piezas principales:
 *   - parseEntry()          parser de los .txt del corpus
 *   - retrieval()            busqueda de la entrada mas relevante
 *   - buildMarkovModel() /
 *     generateFromMarkov()  generador de texto por cadena de Markov
 *   - contexto de conversacion persistido en localStorage
 *
 * Pensado para tocarse a mano: los numeros clave (orden de la
 * cadena de Markov, largo de las respuestas, umbral de retrieval)
 * estan todos arriba, en CONFIG.
 * ============================================================ */

const Humanity = (function () {
  'use strict';

  /* ---------------- configuracion ajustable ---------------- */
  const CONFIG = {
    basePath: 'data/humanity/entries/',
    markovOrder: 2,           // orden de la cadena de Markov (n-gramas)
    transitionMaxWords: 6,    // largo de la frase de transicion generada
    noMatchMaxWords: 32,      // largo de la respuesta cuando no hay retrieval
    snippetSentences: 2,      // cantidad de oraciones que se muestran por extracto
    retrievalMinScore: 3,     // umbral minimo para considerar un match "razonable"
    minTokenLength: 3,        // ignora tokens de consulta mas cortos que esto
  };

  const CONTEXT_KEY = 'humanity_context';
  const MAX_HISTORIAL = 50;

  // palabras muy comunes en español que no aportan nada al retrieval
  // (se filtran solo al analizar la PREGUNTA del usuario, no al entrenar Markov)
  const STOPWORDS = new Set([
    'de', 'la', 'el', 'los', 'las', 'que', 'y', 'a', 'en', 'un', 'una',
    'es', 'se', 'del', 'al', 'lo', 'por', 'con', 'no', 'su', 'para',
    'como', 'mas', 'pero', 'sus', 'le', 'ya', 'o', 'esta', 'este',
    'esa', 'ese', 'esto', 'eso', 'aquello', 'algo', 'nada', 'cada',
    'fue', 'ser', 'hay', 'sobre', 'me', 'te', 'nos', 'sin', 'sino',
    'que', 'quien', 'quienes', 'cual', 'cuales', 'donde', 'cuando',
    'todo', 'toda', 'todos', 'todas', 'otro', 'otra', 'otros', 'otras',
    'muy', 'solo', 'tan', 'asi', 'entre', 'mismo', 'misma', 'tampoco',
  ]);

  // preguntas de seguimiento cortas ("¿y despues?", "¿quien mas?")
  const FOLLOWUP_PATTERN = /^(y|¿y|que paso|qué pasó|quien mas|quien más|y despues|y después|y luego|y entonces|y eso|entonces|y\?)\b/i;

  /* estado interno del modulo */
  const state = {
    entries: [],       // corpus parseado: [{fileName, title, era, tags, body}]
    markovModel: null, // { order, model: Map, starters: [] }
    context: defaultContext(),
    ready: false,
  };

  /* ================================================================
   * PARSER
   * Separa la metadata (TITULO / ERA / TAGS) del cuerpo libre de cada
   * entrada. Si no encuentra metadata, arma valores por defecto sin
   * romper: titulo = nombre de archivo, era = "desconocida", tags = [].
   * ================================================================ */
  function parseEntry(rawText, fileName) {
    const lines = rawText.replace(/\r\n/g, '\n').split('\n');
    const metaLineRe = /^(TITULO|ERA|TAGS)\s*:\s*(.*)$/i;
    const meta = { titulo: null, era: null, tags: null };

    // solo mira las primeras lineas: en cuanto una no matchea el patron
    // de metadata, se corta el bloque y todo lo demas es cuerpo libre
    let i = 0;
    while (i < lines.length) {
      const match = lines[i].match(metaLineRe);
      if (!match) break;
      const key = match[1].toUpperCase();
      const value = match[2].trim();
      if (key === 'TITULO') meta.titulo = value;
      else if (key === 'ERA') meta.era = value;
      else if (key === 'TAGS') meta.tags = value;
      i++;
    }

    // saltea lineas en blanco entre la metadata y el cuerpo
    while (i < lines.length && lines[i].trim() === '') i++;
    const body = lines.slice(i).join('\n').trim();

    const fallbackTitle = fileName.replace(/\.txt$/i, '').replace(/-/g, ' ');
    const tags = meta.tags
      ? meta.tags.split(',').map((t) => t.trim().toLowerCase()).filter(Boolean)
      : [];

    return {
      fileName,
      title: meta.titulo || fallbackTitle,
      era: meta.era || 'desconocida',
      tags,
      body,
    };
  }

  /* Carga index.json y despues cada .txt listado ahi. Si un archivo
   * individual falla, se lo saltea sin tirar abajo el resto del corpus. */
  async function loadCorpus(basePath) {
    const indexRes = await fetch(basePath + 'index.json');
    if (!indexRes.ok) throw new Error('no se pudo leer index.json');
    const fileNames = await indexRes.json();

    const entries = [];
    for (const fileName of fileNames) {
      try {
        const res = await fetch(basePath + fileName);
        if (!res.ok) throw new Error('HTTP ' + res.status);
        const rawText = await res.text();
        entries.push(parseEntry(rawText, fileName));
      } catch (err) {
        console.warn('[HUMANITY] no se pudo cargar', fileName, err);
      }
    }
    return entries;
  }

  /* ================================================================
   * TOKENIZACION / UTILIDADES DE TEXTO
   * ================================================================ */

  // minusculas + sin acentos, para que "explicación" matchee "explicacion"
  function normalize(text) {
    return text
      .toLowerCase()
      .normalize('NFD')
      .replace(/[̀-ͯ]/g, '');
  }

  function tokenize(text) {
    return normalize(text).match(/[a-z0-9]+/g) || [];
  }

  function splitSentences(text) {
    return text.split(/(?<=[.!?])\s+/).filter((s) => s.trim() !== '');
  }

  function capitalizeFirst(text) {
    if (!text) return text;
    return text.charAt(0).toUpperCase() + text.slice(1);
  }

  /* ================================================================
   * RETRIEVAL
   * Busqueda simple por conteo de coincidencias, ponderada:
   * match en tags > match en titulo > match en cuerpo.
   * No es TF-IDF real, pero alcanza para un corpus chico y es facil
   * de ajustar a mano (ver los pesos abajo).
   * ================================================================ */
  function retrieval(query, entries) {
    const queryTokens = tokenize(query).filter(
      (t) => t.length >= CONFIG.minTokenLength && !STOPWORDS.has(t)
    );
    if (queryTokens.length === 0 || entries.length === 0) return null;

    let best = null;
    let bestScore = 0;

    for (const entry of entries) {
      const titleTokens = tokenize(entry.title);
      const bodyTokens = tokenize(entry.body);
      const tagTokens = entry.tags.flatMap((tag) => tokenize(tag));

      let score = 0;
      for (const qt of queryTokens) {
        if (tagTokens.includes(qt)) score += 5;
        if (titleTokens.includes(qt)) score += 3;
        // ocurrencias en el cuerpo, con tope para que una palabra muy
        // repetida no aplaste al resto del scoring
        const bodyMatches = bodyTokens.filter((bt) => bt === qt).length;
        score += Math.min(bodyMatches, 4);
      }

      if (score > bestScore) {
        bestScore = score;
        best = entry;
      }
    }

    if (!best || bestScore < CONFIG.retrievalMinScore) return null;
    return { entry: best, score: bestScore, matchedTokens: queryTokens };
  }

  /* Extrae un fragmento (no la entrada entera) alrededor de la primera
   * oracion que matchea alguno de los tokens buscados. `sentenceOffset`
   * permite "avanzar" en la misma entrada para preguntas de seguimiento
   * tipo "¿y despues?". Devuelve tambien el indice usado, para que el
   * contexto pueda recordar por donde se quedo. */
  function extractSnippet(entry, matchedTokens, sentenceOffset) {
    const sentences = splitSentences(entry.body);
    if (sentences.length === 0) return { text: '', index: 0 };

    let startIndex = 0;
    if (matchedTokens && matchedTokens.length > 0) {
      const found = sentences.findIndex((s) => {
        const st = tokenize(s);
        return matchedTokens.some((t) => st.includes(t));
      });
      if (found !== -1) startIndex = found;
    }

    const index = Math.min(startIndex + (sentenceOffset || 0), sentences.length - 1);
    const end = Math.min(sentences.length, index + CONFIG.snippetSentences);
    return { text: sentences.slice(index, end).join(' '), index };
  }

  /* ================================================================
   * MARKOV
   * Cadena de Markov de orden N (por defecto 2) entrenada con el
   * texto concatenado de todos los cuerpos del corpus. Se reconstruye
   * entera cada vez que se carga el corpus (ver init()), asi que
   * agregar .txt nuevos al index.json alcanza para que el estilo de
   * generacion los incorpore, sin tocar nada de este archivo.
   * ================================================================ */

  // corpusTexts: array de strings (un string por entrada del corpus)
  function buildMarkovModel(corpusTexts, order) {
    order = order || CONFIG.markovOrder;
    const model = new Map(); // "palabra1 palabra2" -> [posibles palabras siguientes]
    const starters = [];      // n-gramas que arrancan una oracion (para no empezar a generar en medio de una frase)

    for (const text of corpusTexts) {
      const words = text.split(/\s+/).filter(Boolean);
      if (words.length <= order) continue;

      for (let i = 0; i <= words.length - order - 1; i++) {
        const gram = words.slice(i, i + order);
        const key = gram.join(' ');
        const nextWord = words[i + order];

        if (!model.has(key)) model.set(key, []);
        model.get(key).push(nextWord);

        const prevWord = i > 0 ? words[i - 1] : null;
        if (i === 0 || (prevWord && /[.!?]$/.test(prevWord))) {
          starters.push(key);
        }
      }
    }

    return {
      order,
      model,
      starters: starters.length > 0 ? starters : Array.from(model.keys()),
    };
  }

  // camina la cadena desde un starter aleatorio hasta maxWords, o hasta
  // encontrar un final de oracion natural (con un minimo de largo).
  function generateFromMarkov(markovModel, maxWords) {
    maxWords = maxWords || 20;
    if (!markovModel || markovModel.starters.length === 0) return '';

    let key = markovModel.starters[Math.floor(Math.random() * markovModel.starters.length)];
    const words = key.split(' ');
    const minWords = Math.min(maxWords, 8);

    while (words.length < maxWords) {
      const candidates = markovModel.model.get(key);
      if (!candidates || candidates.length === 0) break;

      const nextWord = candidates[Math.floor(Math.random() * candidates.length)];
      words.push(nextWord);

      if (/[.!?]$/.test(nextWord) && words.length >= minWords) break;
      key = words.slice(-markovModel.order).join(' ');
    }

    return capitalizeFirst(words.join(' '));
  }

  /* ================================================================
   * CONTEXTO DE CONVERSACION (persistido en localStorage)
   * ================================================================ */
  function defaultContext() {
    return { modo: 'inactivo', ultimaEntradaConsultada: null, historial: [] };
  }

  function loadContext() {
    try {
      const raw = localStorage.getItem(CONTEXT_KEY);
      if (!raw) return defaultContext();
      return Object.assign(defaultContext(), JSON.parse(raw));
    } catch (err) {
      return defaultContext();
    }
  }

  function saveContext() {
    try {
      localStorage.setItem(CONTEXT_KEY, JSON.stringify(state.context));
    } catch (err) {
      // localStorage no disponible (modo privado, cuota llena, etc.):
      // no es critico, simplemente no persiste entre recargas
    }
  }

  function isFollowUp(query) {
    if (!state.context.ultimaEntradaConsultada) return false;
    const trimmed = query.trim();
    const wordCount = tokenize(trimmed).length;
    return wordCount > 0 && wordCount <= 5 && FOLLOWUP_PATTERN.test(trimmed);
  }

  function pushHistorial(item) {
    state.context.historial.push(item);
    if (state.context.historial.length > MAX_HISTORIAL) {
      state.context.historial.shift();
    }
  }

  /* ================================================================
   * RESPUESTAS: combina retrieval (factual) + Markov (generado)
   * ================================================================ */
  function buildFactualAnswer(entry, snippetText) {
    let transition = generateFromMarkov(state.markovModel, CONFIG.transitionMaxWords);
    transition = transition.replace(/[.!?]+$/, '').trim();
    if (!transition) transition = 'Segun lo que queda registrado';
    const header = transition + '... (registro: "' + entry.title + '", era: ' + entry.era + ')';
    return header + '\n\n' + snippetText;
  }

  function buildNoMatchAnswer() {
    const generated = generateFromMarkov(state.markovModel, CONFIG.noMatchMaxWords);
    if (!generated) return 'No queda nada legible sobre eso. El fragmento se perdio.';
    return generated;
  }

  // logica central: dado un texto de pregunta, arma la respuesta y
  // actualiza el contexto (historial + ultima entrada consultada)
  function respondToQuery(query, opts) {
    opts = opts || {};
    let entry = null;
    let snippetInfo = null;
    let usedFollowUp = false;

    if (!opts.forceRetrieval && isFollowUp(query)) {
      const ref = state.context.ultimaEntradaConsultada;
      entry = state.entries.find((e) => e.fileName === ref.fileName) || null;
      if (entry) {
        usedFollowUp = true;
        snippetInfo = extractSnippet(entry, null, (ref.sentenceIndex || 0) + CONFIG.snippetSentences);
      }
    }

    if (!entry) {
      const result = retrieval(query, state.entries);
      if (result) {
        entry = result.entry;
        snippetInfo = extractSnippet(entry, result.matchedTokens, 0);
      }
    }

    let responseText;
    if (entry && snippetInfo && snippetInfo.text) {
      responseText = buildFactualAnswer(entry, snippetInfo.text);
      state.context.ultimaEntradaConsultada = { fileName: entry.fileName, sentenceIndex: snippetInfo.index };
    } else {
      responseText = buildNoMatchAnswer();
      // si no hubo match, se deja ultimaEntradaConsultada como estaba:
      // una pregunta sin respuesta no deberia borrar el hilo anterior
    }

    pushHistorial({ query, fileName: entry ? entry.fileName : null, followUp: usedFollowUp, ts: Date.now() });
    saveContext();

    return responseText.split('\n');
  }

  /* ================================================================
   * COMANDOS DEL MODO HUMANITY
   * ================================================================ */
  function listEntries() {
    if (state.entries.length === 0) {
      return ['No hay entradas cargadas en este archivo.'];
    }
    const lines = ['Indice de fragmentos disponibles:', ''];
    state.entries.forEach((e, idx) => {
      lines.push('  [' + (idx + 1) + '] ' + e.title + '  (era: ' + e.era + ')');
    });
    lines.push('');
    lines.push('Pregunta sobre alguno, o escribi "sobre <tema>".');
    return lines;
  }

  function handleInput(raw) {
    const trimmed = raw.trim();
    if (trimmed === '') return [];

    const lower = normalize(trimmed);
    if (lower === 'listar' || lower === 'indice') {
      return listEntries();
    }

    const sobreMatch = trimmed.match(/^sobre\s+(.+)$/i);
    if (sobreMatch) {
      return respondToQuery(sobreMatch[1], { forceRetrieval: true });
    }

    return respondToQuery(trimmed, {});
  }

  /* ================================================================
   * CICLO DE VIDA DEL MODULO
   * ================================================================ */
  async function init() {
    if (state.ready) return state;

    state.entries = await loadCorpus(CONFIG.basePath);
    state.markovModel = buildMarkovModel(state.entries.map((e) => e.body));
    state.context = loadContext();
    state.ready = true;
    return state;
  }

  function enter() {
    const hadPriorContext = state.context.historial.length > 0;
    state.context.modo = 'humanity';
    saveContext();

    const lines = [];
    if (hadPriorContext) {
      lines.push('retomando conexion con archivo previo...');
      lines.push('');
    }
    lines.push('HUMANITY // sistema de respaldo activo.');
    lines.push('Quedan fragmentos. No todos. Lo que tengo, lo comparto.');
    lines.push('("listar" para ver el indice / "salir" para volver)');
    lines.push('');
    return lines;
  }

  function exit() {
    state.context.modo = 'inactivo';
    saveContext();
    return ['Conexion con HUMANITY suspendida.', ''];
  }

  function isReady() {
    return state.ready;
  }

  /* API publica del modulo */
  return {
    init,
    enter,
    exit,
    handleInput,
    isReady,
    // exportadas para poder ajustarlas/testearlas a mano desde la consola
    parseEntry,
    buildMarkovModel,
    generateFromMarkov,
    retrieval,
    CONFIG,
  };
})();
