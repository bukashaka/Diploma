;; =====================================================================
;;  ИМИТАЦИОННАЯ МОДЕЛЬ МАРШРУТИЗАЦИИ В СЕТИ MANET
;;  Сравнение алгоритмов: DQN-Routing, Q-Routing, AODV, DSDV
;;
;;  Реализованы: расстановка узлов, мобильность Random Waypoint,
;;  радиомодель (two-ray ground) и SNR, энергопотребление, очереди и
;;  упрощённый MAC (бюджет передачи + коллизии), генерация CBR-трафика,
;;  главный цикл go, диспетчер пересылки и сбор всех метрик.
;; =====================================================================

extensions [ table ]

;; ------------------------------- ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ ----------------
globals [
  ;; --- РЕДАКТИРУЕМЫЕ ПАРАМЕТРЫ ---
  field-size              ; размер квадратной области, м
  min-speed               ; диапазон скорости узлов, м/с
  packet-size-bytes       ; размер пакета данных, байт
  initial-energy          ; начальная энергия узла, Дж
  initial-ttl             ; TTL пакета
  queue-capacity          ; ёмкость очереди узла, пакетов
  sim-duration-seconds    ; длительность моделирования, с
  dt                      ; шаг времени (длительность тика), с
  topology-update-interval; период пересчёта связности, с
  tx-budget               ; макс. число пакетов, передаваемых узлом за тик (MAC)
  collision-factor        ; чувствительность к коллизиям (упрощ. CSMA/CA)
  max-neighbors           ; K_max — максимальное число соседей (для DQN)

  ;; --- ГИПЕРПАРАМЕТРЫ RL (общие для агентов) ---
  gamma-rl                ; коэффициент дисконтирования
  epsilon                 ; текущее значение epsilon
  epsilon-min             ; минимальное epsilon
  epsilon-decay           ; коэффициент убывания epsilon
  ql-alpha                ; скорость обучения Q-таблицы
  ql-initial-q            ; начальное значение Q

  ;; --- ПРОИЗВОДНЫЕ КОНСТАНТЫ (вычисляются в setup-globals) ---
  sim-time                ; текущее модельное время, с (= ticks * dt)
  mpp                     ; метров на патч (масштаб области -> мир NetLogo)
  topology-update-ticks   ; период пересчёта связности, тиков
  bandwidth               ; пропускная способность канала, бит/с
  processing-delay        ; задержка обработки на переход, с

  ;; --- РАДИОМОДЕЛЬ (two-ray ground) ---
  tx-power                ; мощность РЧ-передатчика, Вт
  noise-floor             ; уровень шума, Вт
  antenna-height          ; высота антенны, м
  wavelength              ; длина волны, м (зарезервировано)
  snr-max                 ; опорное SNR для нормировки, дБ

  ;; --- ЭНЕРГОПОТРЕБЛЕНИЕ ---
  energy-tx               ; мощность передачи, Вт
  energy-rx               ; мощность приёма, Вт
  energy-idle             ; фоновое потребление, Вт

  ;; --- ФУНКЦИЯ ВОЗНАГРАЖДЕНИЯ ---
  w-delay w-energy w-link w-delivery
  r-success r-drop
  tau-max e-max q-max     ; нормировочные константы

  ;; --- DQN: РАЗДЕЛЯЕМАЯ (parameter-sharing) Q-СЕТЬ ---
  dqn-state-dim           ; размерность вектора состояния
  dqn-action-dim          ; = K_max (число действий)
  dqn-hidden-sizes        ; список размеров скрытых слоёв
  dqn-weights dqn-biases          ; онлайн-сеть: списки матриц/векторов
  dqn-tgt-weights dqn-tgt-biases  ; целевая сеть
  dqn-lr dqn-tau          ; скорость обучения и коэф. мягкого обновления
  dqn-buffer              ; буфер воспроизведения (таблица, циклический)
  dqn-buf-ptr dqn-buf-count       ; указатель записи и заполненность буфера
  dqn-buffer-capacity dqn-batch-size dqn-train-steps
  train-every                     ; обучать раз в N тиков
  dqn-train-enabled?              ; включено ли обучение DQN
  grad-clip                       ; порог отсечения градиента

  ;; --- ГЛОБАЛЬНЫЕ СЧЁТЧИКИ МЕТРИК ---
  pkts-generated          ; всего сгенерировано пакетов данных (N_sent)
  pkts-delivered          ; доставлено (N_received)
  pkts-dropped            ; потеряно
  ctrl-pkts               ; служебные пакеты маршрутизации (N_control)
  sum-delay               ; сумма сквозных задержек доставленных пакетов, с
  max-delay-seen          ; максимальная сквозная задержка, с
  sum-hops                ; сумма переходов доставленных пакетов
  delivered-bits          ; суммарный объём доставленных данных, бит
  sum-route-discovery     ; суммарное время обнаружения маршрутов, с (AODV)
  route-discovery-count   ; число обнаружений маршрутов

  ;; --- КОНСТАНТЫ ПРОТОКОЛОВ AODV / DSDV ---
  aodv-route-lifetime     ; время жизни записи маршрута, с
  aodv-discovery-timeout  ; таймаут поиска маршрута (затем сброс буфера), с
  aodv-max-hops           ; ограничение глубины распространения RREQ
  dsdv-update-interval    ; период периодических обновлений DSDV, с (Часть 6)

  ;; --- ПРОЧЕЕ ---
  next-packet-id          ; счётчик логических идентификаторов пакетов
  cbr-flows               ; список потоков: каждая запись [src-who dst-who]
]

;; -------------------------------- ПОРОДЫ -----------------------------
breed [ nodes node ]
breed [ packets packet ]

nodes-own [
  ;; физика / мобильность
  node-energy             ; остаточная энергия, Дж
  dead?                   ; узел исчерпал энергию
  dest-x dest-y           ; целевая точка Random Waypoint (координаты NetLogo)
  node-speed              ; текущая скорость, м/с
  pause-left              ; остаток времени паузы, с

  ;; канальный уровень / очередь
  queue                   ; очередь исходящих пакетов (список who пакетов)
  out-buffer              ; снимок очереди для обработки за тик
  ctrl-queue              ; очередь служебных сообщений (список сообщений-списков)
  ctrl-out                ; снимок служебной очереди для обработки за тик
  queue-cap               ; ёмкость очереди
  pkts-forwarded          ; счётчик пересланных пакетов

  ;; соседи
  my-neighbors            ; turtle-set текущих соседей

  ;; Q-Routing (Часть 4)
  q-table                 ; table: назначение -> (table: сосед -> q)

  ;; AODV (Часть 5)
  aodv-routes             ; table: назначение -> список [next hop seq expire valid?]
  aodv-seq aodv-rreq-id
  aodv-rreq-seen          ; table подавления дубликатов RREQ
  aodv-pending            ; table: назначение -> список пакетов, ждущих маршрут
  aodv-rreq-time          ; table: назначение -> время отправки RREQ

  ;; DSDV (Часть 6)
  dsdv-table              ; table: назначение -> список [next metric seq install]
  dsdv-seq
  dsdv-last-update        ; время последнего периодического обновления
]

packets-own [
  pkt-id                  ; логический идентификатор
  src dst                 ; who источника и назначения
  carrier                 ; who текущего узла-носителя
  prev-hop                ; who предыдущего узла
  ttl                     ; оставшийся TTL
  hop-count               ; пройдено переходов
  birth-time              ; время генерации, с
  delay-acc               ; накопленная сквозная задержка, с
  size-bits               ; размер, бит
  visited                 ; список посещённых who (анти-петли)
  kind                    ; "data" | "rreq" | "rrep" | "rerr" | "dsdv" | "ack"
  ctrl-dst ctrl-origin    ; поля управляющих пакетов
  ctrl-seq ctrl-hopcount
  last-state last-action  ; сохранённые состояние/действие для обучения RL
]

links-own [
  link-snr                ; текущее SNR канала, дБ
  link-active?            ; активна ли связь
]

;; =====================================================================
;;                              SETUP
;; =====================================================================
to setup
  clear-all
  setup-parameters      ; задать редактируемые параметры
  setup-world           ; масштабировать мир под область field-size
  setup-globals         ; вычислить производные константы
  setup-nodes           ; создать узлы
  setup-flows           ; сформировать CBR-потоки
  setup-dqn             ; (заглушка; полная инициализация — в Части 3)
  reset-metrics         ; обнулить счётчики
  reset-ticks
end

to setup-parameters
  ;; === БАЗОВЫЕ ПАРАМЕТРЫ ===
  set field-size               1000   ; область 1000 x 1000 м
  set min-speed                0      ; серия 1: меняем max-speed 0..40
  set packet-size-bytes        512
  set initial-energy           100    ; Дж
  set initial-ttl              64
  set queue-capacity           50
  set sim-duration-seconds     600
  set dt                       0.1    ; 600 с = 6000 тиков
  set topology-update-interval 1.0
  set tx-budget                10     ; узкое место MAC (пакетов/тик)
  set collision-factor         0.02
  set max-neighbors            8      ; K_max
  set routing-protocol         "DQN-Routing" ; смените здесь и нажмите setup

  ;; === ГИПЕРПАРАМЕТРЫ RL ===
  set gamma-rl                 0.95
  set epsilon                  1.0
  set epsilon-min              0.01
  set epsilon-decay            0.9995
  set ql-alpha                 0.1
  set ql-initial-q             -1.0
  set dqn-lr                   0.001
  set dqn-tau                  0.001
  set dqn-buffer-capacity      10000  ; уменьшено для NetLogo
  set dqn-batch-size           32     ; уменьшено для скорости
  set dqn-hidden-sizes         (list 64 64)  ; уменьшено для NetLogo
  set train-every              10     ; шаг обучения раз в 10 тиков (скорость/качество)
  set dqn-train-enabled?       true
  set grad-clip                1.0    ; отсечение градиента
end

to setup-world
  ;; Область field-size метров отображается на мир NetLogo фикс. разрешения.
  let world-dim 100                 ; 101 x 101 патч
  resize-world 0 world-dim 0 world-dim
  set-patch-size 5
  ask patches [ set pcolor white ]
  set mpp (field-size / world-dim)  ; метров на патч (1000/100 = 10)
end

to setup-globals
  set sim-time 0
  set bandwidth 2000000             ; 2 Мбит/с
  set processing-delay 0.0005       ; 0,5 мс задержки обработки на переход

  ;; радиомодель (two-ray ground)
  set tx-power 0.281
  set noise-floor 1.0E-10
  set antenna-height 1.5
  set wavelength 0.125
  set snr-max 40

  ;; энергопотребление
  set energy-tx 0.66
  set energy-rx 0.395
  set energy-idle 0.05

  ;; функция вознаграждения
  set w-delay 0.4
  set w-energy 0.2
  set w-link 0.2
  set w-delivery 0.2
  set r-success 10
  set r-drop 10
  set tau-max 1.0
  set e-max initial-energy
  set q-max queue-capacity

  ;; производные
  set topology-update-ticks max (list 1 (round (topology-update-interval / dt)))

  ;; константы протоколов AODV / DSDV
  set aodv-route-lifetime 3.0
  set aodv-discovery-timeout 2.0
  set aodv-max-hops 30
  set dsdv-update-interval 5.0

  ;; DQN-структуры
  set dqn-state-dim 0
  set dqn-action-dim max-neighbors
  set dqn-weights []
  set dqn-biases []
  set dqn-tgt-weights []
  set dqn-tgt-biases []
  set dqn-buffer []
  set dqn-train-steps 0
end

to setup-nodes
  set-default-shape nodes "circle"
  set-default-shape packets "dot"
  create-nodes num-nodes [
    setxy random-xcor random-ycor
    set color green
    set size 1.6
    set node-energy initial-energy
    set dead? false
    set pause-left 0
    set-new-waypoint
    set queue []
    set out-buffer []
    set ctrl-queue []
    set ctrl-out []
    set queue-cap queue-capacity
    set pkts-forwarded 0
    set my-neighbors no-turtles
    ;; структуры маршрутизации
    set q-table table:make
    set aodv-routes table:make
    set aodv-seq 0
    set aodv-rreq-id 0
    set aodv-rreq-seen table:make
    set aodv-pending table:make
    set aodv-rreq-time table:make
    set dsdv-table table:make
    set dsdv-seq 0
    table:put dsdv-table who (list who 0 dsdv-seq sim-time)  ; маршрут к самому себе (метрика 0)
    set dsdv-last-update (random-float dsdv-update-interval) ; разнести периодические обновления
  ]
  update-topology
end

to setup-flows
  set cbr-flows []
  let made 0
  let attempts 0
  while [ made < num-flows and attempts < 1000 ] [
    let s [who] of one-of nodes
    let d [who] of one-of nodes
    if (s != d) [
      set cbr-flows lput (list s d) cbr-flows
      set made made + 1
    ]
    set attempts attempts + 1
  ]
end

to setup-dqn
  ;; Разделяемая (parameter-sharing) Q-сеть: ОДНА сеть на все узлы.
  ;; Состояние: 4 глобальных признака + 4 признака на каждого из K_max соседей.
  set dqn-state-dim (4 + 4 * max-neighbors)
  set dqn-action-dim max-neighbors
  ;; список размеров слоёв: вход -> скрытые -> выход
  let sizes (sentence (list dqn-state-dim) dqn-hidden-sizes (list dqn-action-dim))
  set dqn-weights []
  set dqn-biases []
  let l 0
  while [ l < (length sizes - 1) ] [
    let n-in  item l sizes
    let n-out item (l + 1) sizes
    let scale sqrt (2 / n-in)               ; инициализация He
    set dqn-weights lput (init-matrix n-out n-in scale) dqn-weights
    set dqn-biases  lput (n-values n-out [ [i] -> 0 ]) dqn-biases
    set l (l + 1)
  ]
  ;; целевая сеть = копия онлайн-сети
  set dqn-tgt-weights dqn-weights
  set dqn-tgt-biases  dqn-biases
  ;; буфер воспроизведения как циклическая таблица (O(1) на запись)
  set dqn-buffer table:make
  set dqn-buf-ptr 0
  set dqn-buf-count 0
  set dqn-train-steps 0
end

to reset-metrics
  set pkts-generated 0
  set pkts-delivered 0
  set pkts-dropped 0
  set ctrl-pkts 0
  set sum-delay 0
  set max-delay-seen 0
  set sum-hops 0
  set delivered-bits 0
  set sum-route-discovery 0
  set route-discovery-count 0
  set next-packet-id 0
end

;; =====================================================================
;;                               GO
;; =====================================================================
to go
  set sim-time (ticks * dt)
  if (sim-time >= sim-duration-seconds) [ stop ]
  if (count nodes with [ not dead? ] = 0) [ stop ]

  move-nodes                                          ; 1) мобильность
  if (ticks mod topology-update-ticks = 0) [ update-topology ] ; 2) топология
  protocol-periodic                                   ; 3) периодика протоколов
  generate-traffic                                    ; 4) генерация CBR
  forward-packets                                     ; 5) пересылка данных
  process-control                                     ; 6) служебные сообщения (AODV/DSDV)
  consume-idle-energy                                 ; 7) фоновая энергия
  rl-train                                            ; 8) обучение RL (заглушка)
  decay-epsilon                                       ; 9) убывание epsilon
  update-visuals                                      ; 10) визуализация
  tick
end

;; ----------------------- МОБИЛЬНОСТЬ (Random Waypoint) ---------------
to move-nodes
  ask nodes with [ not dead? ] [
    ifelse (pause-left > 0) [
      set pause-left (pause-left - dt)
    ] [
      if (node-speed > 0) [
        let dx1 (dest-x - xcor)
        let dy1 (dest-y - ycor)
        let dist sqrt (dx1 * dx1 + dy1 * dy1)
        let step (node-speed * dt / mpp)
        ifelse (dist <= step) [
          setxy dest-x dest-y
          set pause-left (random-float pause-max)
          set-new-waypoint
        ] [
          set heading (atan dx1 dy1)
          fd step
        ]
      ]
    ]
  ]
end

to set-new-waypoint  ; контекст узла
  set dest-x random-xcor
  set dest-y random-ycor
  set node-speed (min-speed + random-float (max (list 0.0 (max-speed - min-speed))))
end

;; ----------------------- СВЯЗНОСТЬ / СОСЕДИ / SNR --------------------
to update-topology
  ask links [ die ]
  let r2 ((tx-radius / mpp) ^ 2)
  ;; пересчёт множества соседей (евклидово расстояние)
  ask nodes with [ not dead? ] [
    let mx xcor
    let myy ycor
    set my-neighbors other nodes with [
      (not dead?) and (((xcor - mx) ^ 2 + (ycor - myy) ^ 2) <= r2)
    ]
  ]
  ;; визуальные связи (каждое неориентированное ребро — один раз)
  ask nodes with [ not dead? ] [
    create-links-with (my-neighbors with [ who > [who] of myself ])
  ]
  ask links [
    set link-active? true
    set link-snr (compute-snr (link-length * mpp))
    set color gray
    set thickness 0.05
  ]
end

to-report compute-snr [ d-meters ]
  ;; Упрощённая two-ray ground: Pr ~ Pt * ht^2 * hr^2 / d^4
  let d max (list d-meters 1.0)
  let pr (tx-power * (antenna-height ^ 2) * (antenna-height ^ 2) / (d ^ 4))
  let snr-linear (pr / noise-floor)
  report (10 * (log snr-linear 10))   ; дБ
end

to-report node-dist-patches [ a b ]
  report sqrt ((([xcor] of a) - ([xcor] of b)) ^ 2 + (([ycor] of a) - ([ycor] of b)) ^ 2)
end

to-report link-snr-between [ from-id to-id ]
  ;; SNR между двумя узлами (если связь существует), иначе вычислить по расстоянию
  let a node from-id
  let b node to-id
  if (a = nobody or b = nobody) [ report 0 ]
  report compute-snr ((node-dist-patches a b) * mpp)
end

;; ----------------------- ГЕНЕРАЦИЯ ТРАФИКА (CBR) ---------------------
to generate-traffic
  foreach cbr-flows [ flow ->
    let s item 0 flow
    let d item 1 flow
    ;; дробная интенсивность: вероятность генерации за тик = traffic-rate * dt
    if ((random-float 1) < (traffic-rate * dt)) [
      ask node s [
        if (not dead?) [ create-data-packet d ]
      ]
    ]
  ]
end

to create-data-packet [ dest-id ]  ; контекст узла-источника
  ifelse (length queue >= queue-cap) [
    ;; очередь источника переполнена — пакет сгенерирован и сразу потерян
    set pkts-generated (pkts-generated + 1)
    set pkts-dropped (pkts-dropped + 1)
  ] [
    let me-who who
    let my-x xcor
    let my-y ycor
    hatch-packets 1 [
      set pkt-id next-packet-id
      set src me-who
      set dst dest-id
      set carrier me-who
      set prev-hop -1
      set ttl initial-ttl
      set hop-count 0
      set birth-time sim-time
      set delay-acc 0
      set size-bits (packet-size-bytes * 8)
      set visited (list me-who)
      set kind "data"
      set last-state []
      set last-action -1
      set color yellow
      set size 0.7
      set hidden? false
      setxy my-x my-y
      let pw who
      ask node me-who [ set queue lput pw queue ]
    ]
    set next-packet-id (next-packet-id + 1)
    set pkts-generated (pkts-generated + 1)
  ]
end

;; ----------------------- ПЕРЕСЫЛКА ПАКЕТОВ ---------------------------
to forward-packets
  ;; Фаза 1: снять очереди (не более одного перехода на пакет за тик)
  ask nodes with [ not dead? ] [
    set out-buffer queue
    set queue []
  ]
  ;; Фаза 2: обработать снимок с учётом бюджета передачи tx-budget
  ask nodes with [ not dead? ] [
    let budget tx-budget
    while [ (budget > 0) and (not empty? out-buffer) and (not dead?) ] [
      let pw first out-buffer
      set out-buffer but-first out-buffer
      service-packet pw
      set budget (budget - 1)
    ]
    ;; не обработанные из-за бюджета — остаются в очереди до следующего тика
    set queue (sentence out-buffer queue)
    set out-buffer []
  ]
end

to service-packet [ pw ]
  let p packet-by-who pw
  if (p != nobody) [
    ask p [
      ifelse (ttl <= 0) [
        register-drop "ttl"
      ] [
        ifelse (dst = carrier) [
          deliver-packet
        ] [
          let nh choose-next-hop carrier dst self
          ifelse (nh = nobody) [
            handle-no-route carrier
          ] [
            ifelse (nh = -1) [
              ;; пакет буферизован протоколом (AODV) до установления маршрута — ничего не делаем
            ] [
              ifelse (transmission-fails? carrier nh) [
                register-drop "collision"
              ] [
                forward-to carrier nh
              ]
            ]
          ]
        ]
      ]
    ]
  ]
end

;; --- Диспетчер выбора следующего перехода (точка подключения протоколов) ---
to-report choose-next-hop [ here-id dest-id pkt ]
  if (routing-protocol = "DQN-Routing") [ report dqn-next-hop  here-id dest-id pkt ]
  if (routing-protocol = "Q-Routing")   [ report ql-next-hop   here-id dest-id pkt ]
  if (routing-protocol = "AODV")        [ report aodv-next-hop here-id dest-id pkt ]
  if (routing-protocol = "DSDV")        [ report dsdv-next-hop here-id dest-id pkt ]
  report fallback-next-hop here-id dest-id pkt
end

;; --- DQN-Routing ---
to-report dqn-next-hop [ here-id dest-id pkt ]
  let hn node here-id
  if (hn = nobody) [ report nobody ]
  let dn node dest-id
  let hopc [hop-count] of pkt
  ;; построить вектор состояния и слоты соседей
  let st-info dqn-build-state here-id dest-id hopc
  let state    item 0 st-info
  let nbr-whos item 1 st-info
  let n-valid  item 2 st-info
  if (n-valid = 0) [ report nobody ]
  ;; назначение — прямой сосед: выбрать его слот (гарантированная доставка) и записать опыт
  if ((dn != nobody) and (member? dest-id nbr-whos)) [
    let slot (position dest-id nbr-whos)
    ask pkt [ set last-state state  set last-action slot ]
    report dest-id
  ]
  ;; допустимые слоты: соседи, ещё не посещённые (анти-петля)
  let vlist [visited] of pkt
  let valid-slots []
  let s 0
  while [ s < n-valid ] [
    if (not member? (item s nbr-whos) vlist) [ set valid-slots lput s valid-slots ]
    set s (s + 1)
  ]
  if (empty? valid-slots) [ set valid-slots (n-values n-valid [ [i] -> i ]) ] ; выход из тупика
  ;; epsilon-жадный выбор по допустимым слотам
  let chosen-slot 0
  ifelse ((random-float 1) < epsilon) [
    set chosen-slot one-of valid-slots
  ] [
    let qvals nn-forward state dqn-weights dqn-biases
    ;; маскируем недопустимые слоты значением -inf
    let masked (n-values dqn-action-dim
                  [ [i] -> ifelse-value (member? i valid-slots) [ item i qvals ] [ -1.0E10 ] ])
    set chosen-slot argmax-list masked
  ]
  ;; сохранить (s, a) для последующей записи перехода
  ask pkt [ set last-state state  set last-action chosen-slot ]
  report (item chosen-slot nbr-whos)
end
;; =====================================================================
;;                    Q-ROUTING
;; =====================================================================

to-report ql-next-hop [ here-id dest-id pkt ]
  let hn node here-id
  if (hn = nobody) [ report nobody ]
  let dn node dest-id
  let nbrs [my-neighbors] of hn
  if (not any? nbrs) [ report nobody ]
  ;; назначение — прямой сосед: доставить напрямую
  if ((dn != nobody) and (member? dn nbrs)) [
    ask pkt [ set last-action dest-id ]
    report dest-id
  ]
  ;; кандидаты: соседи вне списка посещённых (анти-петля)
  let vlist [visited] of pkt
  let cand nbrs with [ not member? who vlist ]
  if (not any? cand) [ set cand nbrs ]
  ;; epsilon-жадный выбор по Q[dest][neighbor]
  let chosen nobody
  ifelse ((random-float 1) < epsilon) [
    set chosen one-of cand
  ] [
    let best-q -1.0E10
    ask cand [
      let qv (ql-get-q here-id dest-id who)
      if (qv > best-q) [ set best-q qv  set chosen self ]
    ]
    if (chosen = nobody) [ set chosen one-of cand ]
  ]
  ask pkt [ set last-action ([who] of chosen) ]
  report [who] of chosen
end

to-report ql-get-q [ node-id dest neighbor ]
  let qt [q-table] of (node node-id)
  ifelse (table:has-key? qt dest) [
    let sub (table:get qt dest)
    report (ifelse-value (table:has-key? sub neighbor) [ table:get sub neighbor ] [ ql-initial-q ])
  ] [
    report ql-initial-q
  ]
end

to ql-set-q [ node-id dest neighbor value ]
  let qt [q-table] of (node node-id)
  if (not table:has-key? qt dest) [ table:put qt dest table:make ]
  let sub (table:get qt dest)
  table:put sub neighbor value
end

to-report ql-max-q [ node-id dest ]
  let nd node node-id
  let nbr-whos [who] of ([my-neighbors] of nd)
  ifelse (empty? nbr-whos) [ report 0 ] [
    report max (map [ w -> ql-get-q node-id dest w ] nbr-whos)
  ]
end

;; =====================================================================
;;                       AODV
;; =====================================================================

;; --- выбор следующего перехода для пакета данных ---
to-report aodv-next-hop [ here-id dest-id pkt ]
  let hn node here-id
  if (hn = nobody) [ report nobody ]
  if (not any? ([my-neighbors] of hn)) [ report nobody ]       ; изолирован — отбросить
  let nh (aodv-route-nexthop-of here-id dest-id)
  ifelse (nh != nobody) [
    ifelse (member? (node nh) ([my-neighbors] of hn)) [
      report nh                                                ; есть валидный маршрут
    ] [
      aodv-invalidate-route here-id dest-id                    ; следующий узел вне зоны — обрыв
      aodv-start-discovery here-id dest-id pkt
      report -1
    ]
  ] [
    aodv-start-discovery here-id dest-id pkt                   ; маршрута нет — поиск
    report -1
  ]
end

;; --- запрос next-hop из таблицы маршрутов (с учётом срока) ---
to-report aodv-route-nexthop-of [ node-id dest ]
  let rtt [aodv-routes] of (node node-id)
  ifelse (table:has-key? rtt dest) [
    let routing (table:get rtt dest)
    ifelse ((item 4 routing) and (sim-time < (item 3 routing))) [ report (item 0 routing) ] [ report nobody ]
  ] [
    report nobody
  ]
end

;; --- буферизация пакета и инициирование поиска маршрута ---
to aodv-start-discovery [ here-id dest-id pkt ]
  let pw [who] of pkt
  ask node here-id [
    if (not table:has-key? aodv-pending dest-id) [ table:put aodv-pending dest-id [] ]
    table:put aodv-pending dest-id (lput pw (table:get aodv-pending dest-id))
    if (not table:has-key? aodv-rreq-time dest-id) [        ; поиск ещё не запущен
      table:put aodv-rreq-time dest-id sim-time
      aodv-broadcast-new-rreq dest-id
    ]
  ]
end

;; --- источник рассылает новый RREQ ---
to aodv-broadcast-new-rreq [ dest-id ]   ; self = узел-источник
  set aodv-seq (aodv-seq + 1)
  set aodv-rreq-id (aodv-rreq-id + 1)
  table:put aodv-rreq-seen (word who "-" aodv-rreq-id) true   ; игнорировать собственный RREQ
  set ctrl-pkts (ctrl-pkts + 1)
  let msg (list "rreq" who dest-id aodv-seq (aodv-known-dseq dest-id) aodv-rreq-id 0 who)
  ask my-neighbors [ set ctrl-queue lput msg ctrl-queue ]
end

;; --- ретрансляция RREQ ---
to aodv-rebroadcast-rreq [ origin dest-id oseq dseq rid hops ]
  set ctrl-pkts (ctrl-pkts + 1)
  let msg (list "rreq" origin dest-id oseq dseq rid hops who)
  ask my-neighbors [
    set ctrl-queue lput msg ctrl-queue
  ]
end

;; --- отправка/пересылка RREP одному соседу (по обратному пути к источнику) ---
to aodv-emit-rrep [ origin dest-id dseq toward hops ]   ; self = узел
  set ctrl-pkts (ctrl-pkts + 1)
  let msg (list "rrep" origin dest-id 0 dseq 0 hops who)
  if (toward != nobody) [
    let tn (node toward)
    if ((tn != nobody) and (not [dead?] of tn)) [
      ask tn [ set ctrl-queue lput msg ctrl-queue ]
    ]
  ]
end

;; --- обработка очереди служебных сообщений (один переход за тик; AODV и DSDV) ---
to process-control
  if ((routing-protocol = "AODV") or (routing-protocol = "DSDV")) [
    ask nodes with [ not dead? ] [ set ctrl-out ctrl-queue  set ctrl-queue [] ]
    ask nodes with [ not dead? ] [
      foreach ctrl-out [ msg -> handle-ctrl-msg msg ]
      set ctrl-out []
    ]
  ]
end

to handle-ctrl-msg [ msg ]   ; self = принимающий узел
  let k item 0 msg
  if (k = "rreq") [ handle-rreq msg ]
  if (k = "rrep") [ handle-rrep msg ]
  if (k = "dsdv") [ handle-dsdv msg ]
end

;; --- обработка RREQ ---
to handle-rreq [ msg ]   ; self = узел
  let origin item 1 msg
  let dst-w  item 2 msg
  let oseq   item 3 msg
  let dseq   item 4 msg
  let rid    item 5 msg
  let hops   item 6 msg
  let sender item 7 msg
  let key (word origin "-" rid)
  if (not table:has-key? aodv-rreq-seen key) [
    table:put aodv-rreq-seen key true
    aodv-update-route origin sender (hops + 1) oseq          ; обратный маршрут к источнику
    ifelse (who = dst-w) [
      set aodv-seq (max (list aodv-seq dseq))                ; узел-назначение отвечает RREP
      aodv-emit-rrep origin dst-w aodv-seq sender 0
    ] [
      ifelse (aodv-has-fresh-route? dst-w dseq) [             ; промежуточный со свежим маршрутом
        let routing (table:get aodv-routes dst-w)
        aodv-emit-rrep origin dst-w (item 2 routing) sender (item 1 routing)
      ] [
        if ((hops + 1) <= aodv-max-hops) [                    ; иначе ретрансляция
          aodv-rebroadcast-rreq origin dst-w oseq
            (max (list dseq (aodv-known-dseq dst-w))) rid (hops + 1)
        ]
      ]
    ]
  ]
end

;; --- обработка RREP ---
to handle-rrep [ msg ]   ; self = узел
  let origin item 1 msg
  let dst-w  item 2 msg
  let dseq   item 4 msg
  let hops   item 6 msg
  let sender item 7 msg
  aodv-update-route dst-w sender (hops + 1) dseq             ; прямой маршрут к назначению
  ifelse (who = origin) [
    if (table:has-key? aodv-rreq-time dst-w) [               ; RREP достиг источника
      set sum-route-discovery (sum-route-discovery + (sim-time - (table:get aodv-rreq-time dst-w)))
      set route-discovery-count (route-discovery-count + 1)
    ]
    aodv-flush-pending dst-w
  ] [
    let nh (aodv-route-nexthop-of who origin)               ; переслать RREP к источнику
    if (nh != nobody) [ aodv-emit-rrep origin dst-w dseq nh (hops + 1) ]
  ]
end

;; --- обновление записи маршрута с учётом свежести (loop-free по seq) ---
to aodv-update-route [ dest next hops seq ]   ; self = узел
  let do-update? true
  if (table:has-key? aodv-routes dest) [
    let routing (table:get aodv-routes dest)
    if ((seq < (item 2 routing)) or
        ((seq = (item 2 routing)) and ((item 1 routing) <= hops) and (item 4 routing))) [
      set do-update? false
    ]
  ]
  if do-update? [
    table:put aodv-routes dest (list next hops seq (sim-time + aodv-route-lifetime) true)
  ]
end

to-report aodv-known-dseq [ dest ]   ; self = узел
  ifelse (table:has-key? aodv-routes dest) [ report (item 2 (table:get aodv-routes dest)) ]
                                            [ report 0 ]
end

to-report aodv-has-fresh-route? [ dest dseq ]   ; self = узел
  ifelse (table:has-key? aodv-routes dest) [
    let routing (table:get aodv-routes dest)
    report ((item 4 routing) and (sim-time < (item 3 routing)) and ((item 2 routing) >= dseq))
  ] [ report false ]
end

;; --- отправка буферизованных пакетов после установления маршрута ---
to aodv-flush-pending [ dest ]   ; self = узел
  let waitt 0
  if (table:has-key? aodv-rreq-time dest) [ set waitt (sim-time - (table:get aodv-rreq-time dest)) ]
  if (table:has-key? aodv-pending dest) [
    foreach (table:get aodv-pending dest) [ pw ->
      let pp packet-by-who pw
      if (pp != nobody) [
        ifelse (length queue >= queue-cap) [
          ask pp [ register-drop "overflow" ]
        ] [
          ask pp [ set delay-acc (delay-acc + waitt) ]       ; учесть задержку поиска маршрута
          set queue lput pw queue
        ]
      ]
    ]
    table:remove aodv-pending dest
  ]
  if (table:has-key? aodv-rreq-time dest) [ table:remove aodv-rreq-time dest ]
end

;; --- инвалидация маршрута при обрыве (упрощённый RERR) ---
to aodv-invalidate-route [ node-id dest ]
  ask node node-id [
    if (table:has-key? aodv-routes dest) [
      let routing (table:get aodv-routes dest)
      if (item 4 routing) [
        table:put aodv-routes dest (replace-item 4 routing false)
        set ctrl-pkts (ctrl-pkts + 1)                       ; учёт служебного трафика RERR
      ]
    ]
  ]
end

;; --- обслуживание: сброс буфера по таймауту поиска маршрута ---
to aodv-maintenance
  ask nodes with [ not dead? ] [
    foreach (table:keys aodv-rreq-time) [ dest ->
      if ((sim-time - (table:get aodv-rreq-time dest)) > aodv-discovery-timeout) [
        if (table:has-key? aodv-pending dest) [
          foreach (table:get aodv-pending dest) [ pw ->
            let pp packet-by-who pw
            if (pp != nobody) [ ask pp [ register-drop "noroute-timeout" ] ]
          ]
          table:remove aodv-pending dest
        ]
        table:remove aodv-rreq-time dest
      ]
    ]
  ]
end

;; =====================================================================
;;                       DSDV
;; =====================================================================
;; --- выбор следующего перехода для пакета данных ---
to-report dsdv-next-hop [ here-id dest-id pkt ]
  let hn node here-id
  if (hn = nobody) [ report nobody ]
  let ee (dsdv-entry-of here-id dest-id)
  ifelse (ee = false) [
    report nobody
  ] [
    let metr (item 1 ee)
    let nh   (item 0 ee)
    let age  (sim-time - (item 3 ee))
    ifelse ((metr >= 9999) or (age > (3 * dsdv-update-interval))) [
      report nobody                                  ; «бесконечность» или устаревший маршрут
    ] [
      ifelse (member? (node nh) ([my-neighbors] of hn)) [
        report nh
      ] [
        report nobody                                ; next-hop вне зоны — маршрут оборван
      ]
    ]
  ]
end

to-report dsdv-entry-of [ node-id dest ]
  let t [dsdv-table] of (node node-id)
  ifelse (table:has-key? t dest) [ report (table:get t dest) ] [ report false ]
end

;; --- периодический анонс таблицы ---
to dsdv-periodic
  ask nodes with [ not dead? ] [
    if (sim-time - dsdv-last-update >= dsdv-update-interval) [
      set dsdv-last-update sim-time
      set dsdv-seq (dsdv-seq + 2)                            ; собственный seq (чётный)
      table:put dsdv-table who (list who 0 dsdv-seq sim-time); обновить маршрут к себе
      dsdv-broadcast-table
    ]
  ]
end

to dsdv-broadcast-table   ; self = узел
  set ctrl-pkts (ctrl-pkts + 1)                              ; один широковещательный анонс
  let adv []
  foreach (table:keys dsdv-table) [ d ->
    let ee (table:get dsdv-table d)
    set adv lput (list d (item 1 ee) (item 2 ee)) adv          ; [dest metric seq]
  ]
  let msg (list "dsdv" who adv)
  ask my-neighbors [ set ctrl-queue lput msg ctrl-queue ]
end

;; --- приём анонса соседа ---
to handle-dsdv [ msg ]   ; self = узел
  let sender item 1 msg
  let adv    item 2 msg
  foreach adv [ entry ->
    let d    item 0 entry
    let metr item 1 entry
    let sq   item 2 entry
    if (d != who) [
      let nm (ifelse-value (metr >= 9999) [ 9999 ] [ metr + 1 ])  ; ограничить «бесконечность»
      dsdv-consider-route d sender nm sq
    ]
  ]
end

;; --- обновление записи маршрута по правилам свежести DSDV ---
to dsdv-consider-route [ dest next newmetric seq ]   ; self = узел
  let do-update? true
  if (table:has-key? dsdv-table dest) [
    let ee (table:get dsdv-table dest)
    if ((seq < (item 2 ee)) or ((seq = (item 2 ee)) and (newmetric >= (item 1 ee)))) [
      set do-update? false
    ]
  ]
  if do-update? [
    table:put dsdv-table dest (list next newmetric seq sim-time)
  ]
end

;; --- обрыв канала: маршруты через оборванного соседа -> «бесконечность» + триггерный анонс ---
to dsdv-link-break [ node-id broken-next ]
  ask node node-id [
    let changed? false
    foreach (table:keys dsdv-table) [ d ->
      let ee (table:get dsdv-table d)
      if (((item 0 ee) = broken-next) and ((item 1 ee) < 9999)) [
        table:put dsdv-table d (list broken-next 9999 ((item 2 ee) + 1) sim-time)  ; нечётный seq
        set changed? true
      ]
    ]
    if changed? [ dsdv-broadcast-table ]                     ; триггерное обновление
  ]
end

;; Запасной маршрутизатор: жадная географическая пересылка к назначению
to-report fallback-next-hop [ here-id dest-id pkt ]
  let hn node here-id
  ifelse (hn = nobody) [ report nobody ] [
    let nbrs [my-neighbors] of hn
    ifelse (not any? nbrs) [ report nobody ] [
      let dn node dest-id
      ifelse ((dn != nobody) and (member? dn nbrs)) [ report dest-id ] [
        let vlist [visited] of pkt
        let cand nbrs with [ not member? who vlist ]
        if (not any? cand) [ set cand nbrs ]
        ifelse (dn = nobody) [
          report [who] of one-of cand
        ] [
          let best min-one-of cand [ node-dist-patches self dn ]
          report [who] of best
        ]
      ]
    ]
  ]
end

;; Обработка отсутствия маршрута
to handle-no-route [ here-id ]  ; контекст пакета
  register-drop "noroute"
end

;; --- Передача пакета соседу ---
to-report transmission-fails? [ from-id to-id ]
  ;; явный случайный обрыв канала
  if ((random-float 1) < link-failure-prob) [ report true ]
  ;; упрощённая коллизия CSMA/CA: тем вероятнее, чем больше активных соседей
  let contenders 0
  let a node from-id
  if (a != nobody) [
    set contenders [ count (my-neighbors with [ (not dead?) and (not empty? queue) ]) ] of a
  ]
  let p-coll (1 - (1 / (1 + collision-factor * contenders)))
  report ((random-float 1) < p-coll)
end

to forward-to [ from-id to-id ]  ; контекст пакета
  let sender node from-id
  let recv node to-id
  ifelse ((recv = nobody) or ([dead?] of recv)) [
    if (routing-protocol = "AODV") [ aodv-invalidate-route from-id dst ]
    if (routing-protocol = "DSDV") [ dsdv-link-break from-id to-id ]
    register-drop "linkbreak"
  ] [
    ifelse (([length queue] of recv) >= ([queue-cap] of recv)) [
      register-drop "overflow"      ; переполнение очереди приёмника
    ] [
      ;; задержка перехода: обработка + передача + ожидание в очереди приёмника
      let txt (size-bits / bandwidth)
      let qwait (([length queue] of recv) * txt)
      set delay-acc (delay-acc + processing-delay + txt + qwait)
      ;; энергозатраты
      ask sender [ set node-energy (node-energy - (energy-tx * txt))
                   set pkts-forwarded (pkts-forwarded + 1) ]
      ask recv   [ set node-energy (node-energy - (energy-rx * txt)) ]
      ;; обновление полей пакета
      set prev-hop from-id
      set carrier to-id
      set ttl (ttl - 1)
      set hop-count (hop-count + 1)
      if (not member? to-id visited) [ set visited lput to-id visited ]
      ;; постановка в очередь приёмника + визуальное перемещение
      ask recv [ set queue lput ([who] of myself) queue ]
      setxy [xcor] of recv [ycor] of recv
      ;; запись опыта RL до возможной гибели узла
      rl-record-success from-id to-id (processing-delay + txt + qwait)
      ;; проверка исчерпания энергии
      check-energy from-id
      check-energy to-id
    ]
  ]
end

to deliver-packet  ; контекст пакета
  set pkts-delivered (pkts-delivered + 1)
  set sum-delay (sum-delay + delay-acc)
  if (delay-acc > max-delay-seen) [ set max-delay-seen delay-acc ]
  set sum-hops (sum-hops + hop-count)
  set delivered-bits (delivered-bits + size-bits)
  die
end

to register-drop [ reason ]  ; контекст пакета
  rl-record-drop
  set pkts-dropped (pkts-dropped + 1)
  die
end

;; ----------------------- ЭНЕРГИЯ И ОТКАЗЫ УЗЛОВ ----------------------
to consume-idle-energy
  ask nodes with [ not dead? ] [
    set node-energy (node-energy - (energy-idle * dt))
  ]
  ask nodes with [ (not dead?) and (node-energy <= 0) ] [ check-energy who ]
end

to check-energy [ nid ]
  ask node nid [
    if ((not dead?) and (node-energy <= 0)) [
      set node-energy 0
      set dead? true
      set color red
      ;; пакеты в очереди мёртвого узла теряются
      foreach queue [ pw ->
        let pp packet-by-who pw
        if (pp != nobody) [ ask pp [ register-drop "nodedeath" ] ]
      ]
      set queue []
      set out-buffer []
      ask my-links [ die ]
      set my-neighbors no-turtles
    ]
  ]
end

;; ----------------------- ПЕРИОДИКА ПРОТОКОЛОВ / ОБУЧЕНИЕ -------------
to protocol-periodic
  if (routing-protocol = "AODV") [ aodv-maintenance ]
  if (routing-protocol = "DSDV") [ dsdv-periodic ]
end

to rl-train
  if (routing-protocol = "DQN-Routing") [
    if (dqn-train-enabled? and (dqn-buf-count >= dqn-batch-size) and (ticks mod train-every = 0)) [
      dqn-train-batch
    ]
  ]
end

to dqn-push [ s a r s2 done? ]
  table:put dqn-buffer dqn-buf-ptr (list s a r s2 done?)
  set dqn-buf-ptr ((dqn-buf-ptr + 1) mod dqn-buffer-capacity)
  set dqn-buf-count (min (list (dqn-buf-count + 1) dqn-buffer-capacity))
end

to dqn-train-batch
  ;; нулевые аккумуляторы градиентов (по структуре весов)
  let gW (map [ w -> mat-scale w 0 ] dqn-weights)
  let gB (map [ b -> vec-scale b 0 ] dqn-biases)
  let bi 0
  while [ bi < dqn-batch-size ] [
    let tr table:get dqn-buffer (random dqn-buf-count)
    let s     item 0 tr
    let a     item 1 tr
    let r     item 2 tr
    let s2    item 3 tr
    let done? item 4 tr
    ;; целевое значение (Double-DQN)
    let target r
    if (not done?) [
      let a-star  argmax-list (nn-forward s2 dqn-weights dqn-biases)
      let q-tgt   nn-forward s2 dqn-tgt-weights dqn-tgt-biases
      set target (r + gamma-rl * (item a-star q-tgt))
    ]
    ;; прямой проход с кэшем и градиент MSE только по выбранному действию a
    let fc   nn-forward-cache s
    let acts item 0 fc
    let zs   item 1 fc
    let out  last acts
    let dL (n-values dqn-action-dim
              [ [i] -> ifelse-value (i = a) [ 2 * ((item a out) - target) ] [ 0 ] ])
    let grads dqn-backprop acts zs dL
    set gW (map [ [acc g] -> mat-plus acc g ] gW (item 0 grads))
    set gB (map [ [acc g] -> vec-plus acc g ] gB (item 1 grads))
    set bi (bi + 1)
  ]
  ;; усреднить по батчу, ограничить градиент и сделать шаг SGD
  let k (1 / dqn-batch-size)
  let l 0
  while [ l < length dqn-weights ] [
    let gw-l (clip-mat (mat-scale (item l gW) k) grad-clip)
    let gb-l (clip-vec (vec-scale (item l gB) k) grad-clip)
    set dqn-weights (replace-item l dqn-weights
                       (mat-minus (item l dqn-weights) (mat-scale gw-l dqn-lr)))
    set dqn-biases  (replace-item l dqn-biases
                       (vec-minus (item l dqn-biases) (vec-scale gb-l dqn-lr)))
    set l (l + 1)
  ]
  dqn-soft-update                      ; мягкое обновление целевой сети
  set dqn-train-steps (dqn-train-steps + 1)
end

to-report dqn-backprop [ acts zs dL ]
  ;; acts = [a0..aL], zs = [z1..zL], dL = dL/dz выходного (линейного) слоя
  let n (length dqn-weights)
  let gWs (n-values n [ [i] -> 0 ])
  let gBs (n-values n [ [i] -> 0 ])
  let delta dL
  let l (n - 1)
  while [ l >= 0 ] [
    set gWs (replace-item l gWs (outer delta (item l acts)))
    set gBs (replace-item l gBs delta)
    if (l > 0) [
      let wT-delta (matT-vec (item l dqn-weights) delta)
      let z-prev   (item (l - 1) zs)
      set delta (map [ [g zz] -> g * (ifelse-value (zz > 0) [ 1 ] [ 0 ]) ] wT-delta z-prev)
    ]
    set l (l - 1)
  ]
  report (list gWs gBs)
end

to dqn-soft-update
  ;; theta_target <- tau * theta_online + (1 - tau) * theta_target
  set dqn-tgt-weights (map [ [wt wo] ->
        (map [ [routing ro] ->
            (map [ [xt xo] -> dqn-tau * xo + (1 - dqn-tau) * xt ] routing ro) ] wt wo) ]
        dqn-tgt-weights dqn-weights)
  set dqn-tgt-biases (map [ [bt bo] ->
        (map [ [xt xo] -> dqn-tau * xo + (1 - dqn-tau) * xt ] bt bo) ]
        dqn-tgt-biases dqn-biases)
end

to decay-epsilon
  set epsilon (max (list epsilon-min (epsilon * epsilon-decay)))
end

;; ----------------------- ВИЗУАЛИЗАЦИЯ --------------------------------
to update-visuals
  ;; пакеты отображаются у своего носителя
  ask packets [
    let c node carrier
    if (c != nobody) [ setxy [xcor] of c [ycor] of c ]
  ]
  ;; цвет живых узлов по остаточной энергии
  ask nodes with [ not dead? ] [
    set color green + 2 * (node-energy / initial-energy)
  ]
end

;; ----------------------- СЛУЖЕБНЫЕ ФУНКЦИИ ВОЗНАГРАЖДЕНИЯ -------------
;; используется алгоритмами RL в Частях 3-4
to-report compute-reward [ delay-s neighbor-energy snr-db delivered? dropped? ]
  let r-d (- (min (list (delay-s / tau-max) 1.0)))
  let r-e (min (list (neighbor-energy / e-max) 1.0))
  let r-l (max (list 0 (min (list (snr-db / snr-max) 1.0))))
  let r-p 0
  if delivered? [ set r-p r-success ]
  if dropped?   [ set r-p (- r-drop) ]
  report (w-delay * r-d + w-energy * r-e + w-link * r-l + w-delivery * r-p)
end

;; =====================================================================
;;             НЕЙРОСЕТЬ DQN НА СПИСКАХ NetLogo
;; =====================================================================
;; Матрица = список строк; строка j матрицы W — веса от всех входов к выходу j.

to-report dot-prod [ u v ]
  let s 0
  (foreach u v [ [a b] -> set s (s + a * b) ])
  report s
end

to-report mat-vec [ m v ]                 ; W * v
  report map [ row -> dot-prod row v ] m
end

to-report matT-vec [ m v ]                ; W^T * v  (v длины n_out -> длины n_in)
  let n-in length (item 0 m)
  let res (n-values n-in [ [i] -> 0 ])
  let j 0
  while [ j < length m ] [
    let row item j m
    let vj  item j v
    set res (map [ [ri rv] -> rv + vj * ri ] row res)   ; res[i] += v[j] * W[j][i]
    set j (j + 1)
  ]
  report res
end

to-report vec-plus  [ u v ] report (map [ [a b] -> a + b ] u v) end
to-report vec-minus [ u v ] report (map [ [a b] -> a - b ] u v) end
to-report vec-scale [ v k ] report (map [ x -> x * k ] v) end
to-report relu-vec  [ v ] report (map [ x -> ifelse-value (x > 0) [ x ] [ 0 ] ] v) end

to-report mat-plus  [ m1 m2 ] report (map [ [r1 r2] -> vec-plus  r1 r2 ] m1 m2) end
to-report mat-minus [ m1 m2 ] report (map [ [r1 r2] -> vec-minus r1 r2 ] m1 m2) end
to-report mat-scale [ m k ]   report (map [ row -> vec-scale row k ] m) end

to-report outer [ d a ]                   ; d (x) a -> матрица |d| x |a|
  report map [ dj -> map [ ai -> dj * ai ] a ] d
end

to-report clip-vec [ v c ] report (map [ x -> (max (list (- c) (min (list c x)))) ] v) end
to-report clip-mat [ m c ] report (map [ row -> clip-vec row c ] m) end

to-report argmax-list [ lst ]
  let best-i 0
  let best-v item 0 lst
  let i 1
  while [ i < length lst ] [
    if (item i lst > best-v) [ set best-v (item i lst)  set best-i i ]
    set i (i + 1)
  ]
  report best-i
end

to-report init-matrix [ rows cols scale ]
  report n-values rows [ [r] -> n-values cols [ [c] -> (random-normal 0 1) * scale ] ]
end

;; прямой проход (ReLU на скрытых слоях, линейный выход)
to-report nn-forward [ x weights biases ]
  let a x
  let n length weights
  let l 0
  while [ l < n ] [
    let z (vec-plus (mat-vec (item l weights) a) (item l biases))
    ifelse (l < (n - 1)) [ set a relu-vec z ] [ set a z ]
    set l (l + 1)
  ]
  report a
end

;; прямой проход с сохранением активаций и пред-активаций (для обратного распространения)
to-report nn-forward-cache [ x ]
  let acts (list x)
  let zs []
  let a x
  let n length dqn-weights
  let l 0
  while [ l < n ] [
    let z (vec-plus (mat-vec (item l dqn-weights) a) (item l dqn-biases))
    set zs lput z zs
    ifelse (l < (n - 1)) [ set a relu-vec z ] [ set a z ]
    set acts lput a acts
    set l (l + 1)
  ]
  report (list acts zs)
end

;; --- Построение вектора состояния ---
;; Возвращает [вектор-состояния, список-who-соседей-по-слотам, число-валидных-слотов].
;; Признаки нормированы в [0,1] (или [-1,1] для направления к назначению).
to-report dqn-build-state [ here-id dest-id hopcount ]
  let hn node here-id
  let dn node dest-id
  let nbr-agents sort ([my-neighbors] of hn)          ; сортировка по who (детерминизм)
  if (length nbr-agents > max-neighbors) [ set nbr-agents sublist nbr-agents 0 max-neighbors ]
  let nbr-whos (map [ a -> [who] of a ] nbr-agents)
  ;; глобальные признаки: направление и расстояние к назначению, доля пройденного пути
  let here-x [xcor] of hn
  let here-y [ycor] of hn
  let dx1 0  let dy1 0  let ddist 0
  if (dn != nobody) [
    set dx1 (([xcor] of dn - here-x) / world-width)
    set dy1 (([ycor] of dn - here-y) / world-height)
    set ddist ((node-dist-patches hn dn) / world-width)
  ]
  let global-feats (list dx1 dy1 (min (list ddist 1)) (min (list (hopcount / initial-ttl) 1)))
  ;; признаки соседей: по 4 на слот (очередь, энергия, SNR, прогресс к назначению)
  let feats []
  let s 0
  while [ s < max-neighbors ] [
    ifelse (s < length nbr-agents) [
      let nb item s nbr-agents
      let q-norm ((length [queue] of nb) / [queue-cap] of nb)
      let e-norm   (([node-energy] of nb) / e-max)
      let snr-norm ((compute-snr ((node-dist-patches hn nb) * mpp)) / snr-max)
      let prog 1
      if (dn != nobody) [ set prog ((node-dist-patches nb dn) / world-width) ]
      set feats (sentence feats (list (min (list q-norm 1))
                                       (min (list e-norm 1))
                                       (max (list 0 (min (list snr-norm 1))))
                                       (min (list prog 1))))
    ] [
      set feats (sentence feats (list 0 0 0 0))       ; пустой слот
    ]
    set s (s + 1)
  ]
  report (list (sentence global-feats feats) nbr-whos (length nbr-agents))
end

;; --- Запись опыта (вызывается из forward-to / register-drop; контекст пакета) ---
to rl-record-success [ from-id to-id hop-delay ]
  if (routing-protocol = "DQN-Routing") [
    if (last-action >= 0) [
      let nb-energy [node-energy] of (node to-id)
      let snr (link-snr-between from-id to-id)
      let done? (to-id = dst)
      let r (compute-reward hop-delay nb-energy snr done? false)
      let s2 (item 0 (dqn-build-state to-id dst hop-count))   ; состояние следующего узла
      dqn-push last-state last-action r s2 done?
      set last-action -1
    ]
  ]
  if (routing-protocol = "Q-Routing") [
    if (last-action >= 0) [
      let nb-energy [node-energy] of (node to-id)
      let snr (link-snr-between from-id to-id)
      let done? (to-id = dst)
      let r (compute-reward hop-delay nb-energy snr done? false)
      let future (ifelse-value done? [ 0 ] [ ql-max-q to-id dst ])  ; max_k Q_j(d,k) у соседа
      let oldq (ql-get-q from-id dst to-id)
      ql-set-q from-id dst to-id (oldq + ql-alpha * (r + gamma-rl * future - oldq))
      set last-action -1
    ]
  ]
end

to rl-record-drop
  if (routing-protocol = "DQN-Routing") [
    if (last-action >= 0) [
      let r (compute-reward tau-max 0 0 false true)          ; макс. штраф + штраф за потерю
      let zero-state (n-values dqn-state-dim [ [i] -> 0 ])
      dqn-push last-state last-action r zero-state true
      set last-action -1
    ]
  ]
  if (routing-protocol = "Q-Routing") [
    if (last-action >= 0) [
      ;; при потере: carrier — это узел-инициатор перехода, last-action — выбранный сосед
      let r (compute-reward tau-max 0 0 false true)
      let oldq (ql-get-q carrier dst last-action)
      ql-set-q carrier dst last-action (oldq + ql-alpha * (r - oldq))   ; терминал: цель = r
      set last-action -1
    ]
  ]
end

;; ----------------------- УТИЛИТЫ -------------------------------------
to-report packet-by-who [ w ]
  let t turtle w
  ifelse ((t != nobody) and ([breed] of t = packets)) [ report t ] [ report nobody ]
end

;; =====================================================================
;;                       РЕПОРТЕРЫ МЕТРИК (для мониторов)
;; =====================================================================
to-report PDR
  ifelse (pkts-generated = 0) [ report 0 ] [ report (100 * pkts-delivered / pkts-generated) ]
end

to-report avg-delay-ms
  ifelse (pkts-delivered = 0) [ report 0 ] [ report (1000 * sum-delay / pkts-delivered) ]
end

to-report max-delay-ms
  report (1000 * max-delay-seen)
end

to-report NRO
  ifelse (pkts-delivered = 0) [ report 0 ] [ report (ctrl-pkts / pkts-delivered) ]
end

to-report throughput-bps
  ifelse (sim-time = 0) [ report 0 ] [ report (delivered-bits / sim-time) ]
end

to-report avg-residual-energy
  ifelse (count nodes = 0) [ report 0 ] [ report (mean [node-energy] of nodes) ]
end

to-report avg-hop-count
  ifelse (pkts-delivered = 0) [ report 0 ] [ report (sum-hops / pkts-delivered) ]
end

to-report packet-loss-rate
  ifelse (pkts-generated = 0) [ report 0 ] [ report (100 * pkts-dropped / pkts-generated) ]
end

to-report control-packet-count
  report ctrl-pkts
end

to-report avg-route-discovery-ms
  ifelse (route-discovery-count = 0) [ report 0 ]
                                     [ report (1000 * sum-route-discovery / route-discovery-count) ]
end

to-report network-utilization
  ifelse (bandwidth = 0) [ report 0 ] [ report (throughput-bps / bandwidth) ]
end

to-report packets-in-flight
  report count packets
end

to-report alive-nodes
  report count nodes with [ not dead? ]
end
@#$#@#$#@
GRAPHICS-WINDOW
440
10
953
524
-1
-1
5.0
1
10
1
1
1
0
1
1
1
0
100
0
100
0
0
1
ticks
30.0

BUTTON
983
43
1046
76
NIL
setup
NIL
1
T
OBSERVER
NIL
NIL
NIL
NIL
1

BUTTON
1062
42
1125
75
NIL
go
T
1
T
OBSERVER
NIL
NIL
NIL
NIL
1

CHOOSER
982
108
1120
153
routing-protocol
routing-protocol
"DQN-Routing" "Q-Routing" "AODV" "DSDV"
0

SLIDER
983
168
1155
201
num-nodes
num-nodes
10
100
50.0
5
1
NIL
HORIZONTAL

SLIDER
983
209
1155
242
tx-radius
tx-radius
50
400
250.0
25
1
NIL
HORIZONTAL

SLIDER
984
250
1156
283
max-speed
max-speed
0
40
20.0
5
1
NIL
HORIZONTAL

SLIDER
984
288
1156
321
pause-max
pause-max
0
30
10.0
1
1
NIL
HORIZONTAL

SLIDER
985
325
1157
358
num-flows
num-flows
1
30
10.0
1
1
NIL
HORIZONTAL

SLIDER
986
361
1158
394
traffic-rate
traffic-rate
1
30
4.0
1
1
NIL
HORIZONTAL

SLIDER
986
398
1158
431
link-failure-prob
link-failure-prob
0
0.5
0.0
0.01
1
NIL
HORIZONTAL

MONITOR
311
482
373
527
PDR (%)
PDR
17
1
11

MONITOR
216
482
304
527
 Потери (%) 
packet-loss-rate
17
1
11

MONITOR
307
324
426
369
Ср. задержка (мс)
avg-delay-ms
17
1
11

MONITOR
146
430
279
475
Макс. задержка (мс)
max-delay-ms
17
1
11

MONITOR
292
377
413
422
Throughput (бит/с)
throughput-bps
17
1
11

MONITOR
292
432
404
477
Утилизация сети
network-utilization
17
1
11

MONITOR
80
531
137
576
NRO
NRO
17
1
11

MONITOR
7
430
140
475
Служебных пакетов
control-packet-count
17
1
11

MONITOR
7
324
141
369
Поиск маршрута (мс)
avg-route-discovery-ms
17
1
11

MONITOR
155
324
291
369
Ср. число переходов
avg-hop-count
17
1
11

MONITOR
7
379
127
424
Ост. энергия (Дж)
avg-residual-energy
17
1
11

MONITOR
117
482
209
527
Живых узлов
alive-nodes
17
1
11

MONITOR
143
532
200
577
epsilon
epsilon
17
1
11

MONITOR
138
378
277
423
Шагов обучения DQN
dqn-train-steps
17
1
11

MONITOR
7
481
111
526
Пакетов в пути
packets-in-flight
17
1
11

MONITOR
7
531
73
576
Время (с)
sim-time
17
1
11

PLOT
1
11
201
161
PDR (%)
Время (тики)
PDR, %
0.0
100.0
0.0
100.0
true
true
"" ""
PENS
"PDR" 1.0 0 -16777216 true "" "plot PDR"

PLOT
199
10
399
160
Средняя задержка (мс)
Время (тики)
мс
0.0
100.0
0.0
50.0
true
true
"" ""
PENS
"delay" 1.0 0 -16777216 true "" "plot avg-delay-ms"

PLOT
0
159
200
309
Пропускная способность (бит/с) 
Время (тики)
бит/с
0.0
100.0
0.0
100000.0
true
false
"" ""
PENS
"thru" 1.0 0 -16777216 true "" "plot throughput-bps"

PLOT
199
159
399
309
Средняя остаточная энергия (Дж)
Время (тики)
Дж
0.0
100.0
0.0
100.0
true
false
"" ""
PENS
"energy" 1.0 0 -16777216 true "" "plot avg-residual-energy"

@#$#@#$#@
## WHAT IS IT?

(a general understanding of what the model is trying to show or explain)

## HOW IT WORKS

(what rules the agents use to create the overall behavior of the model)

## HOW TO USE IT

(how to use the model, including a description of each of the items in the Interface tab)

## THINGS TO NOTICE

(suggested things for the user to notice while running the model)

## THINGS TO TRY

(suggested things for the user to try to do (move sliders, switches, etc.) with the model)

## EXTENDING THE MODEL

(suggested things to add or change in the Code tab to make the model more complicated, detailed, accurate, etc.)

## NETLOGO FEATURES

(interesting or unusual features of NetLogo that the model uses, particularly in the Code tab; or where workarounds were needed for missing features)

## RELATED MODELS

(models in the NetLogo Models Library and elsewhere which are of related interest)

## CREDITS AND REFERENCES

(a reference to the model's URL on the web if it has one, as well as any other necessary credits, citations, and links)
@#$#@#$#@
default
true
0
Polygon -7500403 true true 150 5 40 250 150 205 260 250

airplane
true
0
Polygon -7500403 true true 150 0 135 15 120 60 120 105 15 165 15 195 120 180 135 240 105 270 120 285 150 270 180 285 210 270 165 240 180 180 285 195 285 165 180 105 180 60 165 15

arrow
true
0
Polygon -7500403 true true 150 0 0 150 105 150 105 293 195 293 195 150 300 150

box
false
0
Polygon -7500403 true true 150 285 285 225 285 75 150 135
Polygon -7500403 true true 150 135 15 75 150 15 285 75
Polygon -7500403 true true 15 75 15 225 150 285 150 135
Line -16777216 false 150 285 150 135
Line -16777216 false 150 135 15 75
Line -16777216 false 150 135 285 75

bug
true
0
Circle -7500403 true true 96 182 108
Circle -7500403 true true 110 127 80
Circle -7500403 true true 110 75 80
Line -7500403 true 150 100 80 30
Line -7500403 true 150 100 220 30

butterfly
true
0
Polygon -7500403 true true 150 165 209 199 225 225 225 255 195 270 165 255 150 240
Polygon -7500403 true true 150 165 89 198 75 225 75 255 105 270 135 255 150 240
Polygon -7500403 true true 139 148 100 105 55 90 25 90 10 105 10 135 25 180 40 195 85 194 139 163
Polygon -7500403 true true 162 150 200 105 245 90 275 90 290 105 290 135 275 180 260 195 215 195 162 165
Polygon -16777216 true false 150 255 135 225 120 150 135 120 150 105 165 120 180 150 165 225
Circle -16777216 true false 135 90 30
Line -16777216 false 150 105 195 60
Line -16777216 false 150 105 105 60

car
false
0
Polygon -7500403 true true 300 180 279 164 261 144 240 135 226 132 213 106 203 84 185 63 159 50 135 50 75 60 0 150 0 165 0 225 300 225 300 180
Circle -16777216 true false 180 180 90
Circle -16777216 true false 30 180 90
Polygon -16777216 true false 162 80 132 78 134 135 209 135 194 105 189 96 180 89
Circle -7500403 true true 47 195 58
Circle -7500403 true true 195 195 58

circle
false
0
Circle -7500403 true true 0 0 300

circle 2
false
0
Circle -7500403 true true 0 0 300
Circle -16777216 true false 30 30 240

cow
false
0
Polygon -7500403 true true 200 193 197 249 179 249 177 196 166 187 140 189 93 191 78 179 72 211 49 209 48 181 37 149 25 120 25 89 45 72 103 84 179 75 198 76 252 64 272 81 293 103 285 121 255 121 242 118 224 167
Polygon -7500403 true true 73 210 86 251 62 249 48 208
Polygon -7500403 true true 25 114 16 195 9 204 23 213 25 200 39 123

cylinder
false
0
Circle -7500403 true true 0 0 300

dot
false
0
Circle -7500403 true true 90 90 120

face happy
false
0
Circle -7500403 true true 8 8 285
Circle -16777216 true false 60 75 60
Circle -16777216 true false 180 75 60
Polygon -16777216 true false 150 255 90 239 62 213 47 191 67 179 90 203 109 218 150 225 192 218 210 203 227 181 251 194 236 217 212 240

face neutral
false
0
Circle -7500403 true true 8 7 285
Circle -16777216 true false 60 75 60
Circle -16777216 true false 180 75 60
Rectangle -16777216 true false 60 195 240 225

face sad
false
0
Circle -7500403 true true 8 8 285
Circle -16777216 true false 60 75 60
Circle -16777216 true false 180 75 60
Polygon -16777216 true false 150 168 90 184 62 210 47 232 67 244 90 220 109 205 150 198 192 205 210 220 227 242 251 229 236 206 212 183

fish
false
0
Polygon -1 true false 44 131 21 87 15 86 0 120 15 150 0 180 13 214 20 212 45 166
Polygon -1 true false 135 195 119 235 95 218 76 210 46 204 60 165
Polygon -1 true false 75 45 83 77 71 103 86 114 166 78 135 60
Polygon -7500403 true true 30 136 151 77 226 81 280 119 292 146 292 160 287 170 270 195 195 210 151 212 30 166
Circle -16777216 true false 215 106 30

flag
false
0
Rectangle -7500403 true true 60 15 75 300
Polygon -7500403 true true 90 150 270 90 90 30
Line -7500403 true 75 135 90 135
Line -7500403 true 75 45 90 45

flower
false
0
Polygon -10899396 true false 135 120 165 165 180 210 180 240 150 300 165 300 195 240 195 195 165 135
Circle -7500403 true true 85 132 38
Circle -7500403 true true 130 147 38
Circle -7500403 true true 192 85 38
Circle -7500403 true true 85 40 38
Circle -7500403 true true 177 40 38
Circle -7500403 true true 177 132 38
Circle -7500403 true true 70 85 38
Circle -7500403 true true 130 25 38
Circle -7500403 true true 96 51 108
Circle -16777216 true false 113 68 74
Polygon -10899396 true false 189 233 219 188 249 173 279 188 234 218
Polygon -10899396 true false 180 255 150 210 105 210 75 240 135 240

house
false
0
Rectangle -7500403 true true 45 120 255 285
Rectangle -16777216 true false 120 210 180 285
Polygon -7500403 true true 15 120 150 15 285 120
Line -16777216 false 30 120 270 120

leaf
false
0
Polygon -7500403 true true 150 210 135 195 120 210 60 210 30 195 60 180 60 165 15 135 30 120 15 105 40 104 45 90 60 90 90 105 105 120 120 120 105 60 120 60 135 30 150 15 165 30 180 60 195 60 180 120 195 120 210 105 240 90 255 90 263 104 285 105 270 120 285 135 240 165 240 180 270 195 240 210 180 210 165 195
Polygon -7500403 true true 135 195 135 240 120 255 105 255 105 285 135 285 165 240 165 195

line
true
0
Line -7500403 true 150 0 150 300

line half
true
0
Line -7500403 true 150 0 150 150

pentagon
false
0
Polygon -7500403 true true 150 15 15 120 60 285 240 285 285 120

person
false
0
Circle -7500403 true true 110 5 80
Polygon -7500403 true true 105 90 120 195 90 285 105 300 135 300 150 225 165 300 195 300 210 285 180 195 195 90
Rectangle -7500403 true true 127 79 172 94
Polygon -7500403 true true 195 90 240 150 225 180 165 105
Polygon -7500403 true true 105 90 60 150 75 180 135 105

plant
false
0
Rectangle -7500403 true true 135 90 165 300
Polygon -7500403 true true 135 255 90 210 45 195 75 255 135 285
Polygon -7500403 true true 165 255 210 210 255 195 225 255 165 285
Polygon -7500403 true true 135 180 90 135 45 120 75 180 135 210
Polygon -7500403 true true 165 180 165 210 225 180 255 120 210 135
Polygon -7500403 true true 135 105 90 60 45 45 75 105 135 135
Polygon -7500403 true true 165 105 165 135 225 105 255 45 210 60
Polygon -7500403 true true 135 90 120 45 150 15 180 45 165 90

sheep
false
15
Circle -1 true true 203 65 88
Circle -1 true true 70 65 162
Circle -1 true true 150 105 120
Polygon -7500403 true false 218 120 240 165 255 165 278 120
Circle -7500403 true false 214 72 67
Rectangle -1 true true 164 223 179 298
Polygon -1 true true 45 285 30 285 30 240 15 195 45 210
Circle -1 true true 3 83 150
Rectangle -1 true true 65 221 80 296
Polygon -1 true true 195 285 210 285 210 240 240 210 195 210
Polygon -7500403 true false 276 85 285 105 302 99 294 83
Polygon -7500403 true false 219 85 210 105 193 99 201 83

square
false
0
Rectangle -7500403 true true 30 30 270 270

square 2
false
0
Rectangle -7500403 true true 30 30 270 270
Rectangle -16777216 true false 60 60 240 240

star
false
0
Polygon -7500403 true true 151 1 185 108 298 108 207 175 242 282 151 216 59 282 94 175 3 108 116 108

target
false
0
Circle -7500403 true true 0 0 300
Circle -16777216 true false 30 30 240
Circle -7500403 true true 60 60 180
Circle -16777216 true false 90 90 120
Circle -7500403 true true 120 120 60

tree
false
0
Circle -7500403 true true 118 3 94
Rectangle -6459832 true false 120 195 180 300
Circle -7500403 true true 65 21 108
Circle -7500403 true true 116 41 127
Circle -7500403 true true 45 90 120
Circle -7500403 true true 104 74 152

triangle
false
0
Polygon -7500403 true true 150 30 15 255 285 255

triangle 2
false
0
Polygon -7500403 true true 150 30 15 255 285 255
Polygon -16777216 true false 151 99 225 223 75 224

truck
false
0
Rectangle -7500403 true true 4 45 195 187
Polygon -7500403 true true 296 193 296 150 259 134 244 104 208 104 207 194
Rectangle -1 true false 195 60 195 105
Polygon -16777216 true false 238 112 252 141 219 141 218 112
Circle -16777216 true false 234 174 42
Rectangle -7500403 true true 181 185 214 194
Circle -16777216 true false 144 174 42
Circle -16777216 true false 24 174 42
Circle -7500403 false true 24 174 42
Circle -7500403 false true 144 174 42
Circle -7500403 false true 234 174 42

turtle
true
0
Polygon -10899396 true false 215 204 240 233 246 254 228 266 215 252 193 210
Polygon -10899396 true false 195 90 225 75 245 75 260 89 269 108 261 124 240 105 225 105 210 105
Polygon -10899396 true false 105 90 75 75 55 75 40 89 31 108 39 124 60 105 75 105 90 105
Polygon -10899396 true false 132 85 134 64 107 51 108 17 150 2 192 18 192 52 169 65 172 87
Polygon -10899396 true false 85 204 60 233 54 254 72 266 85 252 107 210
Polygon -7500403 true true 119 75 179 75 209 101 224 135 220 225 175 261 128 261 81 224 74 135 88 99

wheel
false
0
Circle -7500403 true true 3 3 294
Circle -16777216 true false 30 30 240
Line -7500403 true 150 285 150 15
Line -7500403 true 15 150 285 150
Circle -7500403 true true 120 120 60
Line -7500403 true 216 40 79 269
Line -7500403 true 40 84 269 221
Line -7500403 true 40 216 269 79
Line -7500403 true 84 40 221 269

wolf
false
0
Polygon -16777216 true false 253 133 245 131 245 133
Polygon -7500403 true true 2 194 13 197 30 191 38 193 38 205 20 226 20 257 27 265 38 266 40 260 31 253 31 230 60 206 68 198 75 209 66 228 65 243 82 261 84 268 100 267 103 261 77 239 79 231 100 207 98 196 119 201 143 202 160 195 166 210 172 213 173 238 167 251 160 248 154 265 169 264 178 247 186 240 198 260 200 271 217 271 219 262 207 258 195 230 192 198 210 184 227 164 242 144 259 145 284 151 277 141 293 140 299 134 297 127 273 119 270 105
Polygon -7500403 true true -1 195 14 180 36 166 40 153 53 140 82 131 134 133 159 126 188 115 227 108 236 102 238 98 268 86 269 92 281 87 269 103 269 113

x
false
0
Polygon -7500403 true true 270 75 225 30 30 225 75 270
Polygon -7500403 true true 30 75 75 30 270 225 225 270
@#$#@#$#@
NetLogo 6.4.0
@#$#@#$#@
@#$#@#$#@
@#$#@#$#@
@#$#@#$#@
@#$#@#$#@
default
0.0
-0.2 0 0.0 1.0
0.0 1 1.0 0.0
0.2 0 0.0 1.0
link direction
true
0
Line -7500403 true 150 150 90 180
Line -7500403 true 150 150 210 180
@#$#@#$#@
0
@#$#@#$#@
