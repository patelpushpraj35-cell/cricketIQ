(() => {
  const CATEGORY_KEYS = [
    'traditional',
    'pressure',
    'impact',
    'important_matches',
    'consistency',
    'game_intelligence'
  ];
  const CATEGORY_LABELS = {
    traditional: 'Traditional statistics',
    pressure: 'Match situation & pressure',
    impact: 'High impact & match winning',
    important_matches: 'Important matches',
    consistency: 'Consistency & adaptability',
    game_intelligence: 'Game intelligence & fielding'
  };
  const MASTER_LABELS = {
    consistency_score: 'Consistency',
    pressure_score: 'Pressure',
    match_impact_score: 'Match impact'
  };
  const METRIC_LABELS = {
    bat_runs_per_match: 'Runs per IPL appearance',
    bat_average: 'Batting average',
    bat_strike_rate: 'Batting strike rate',
    bat_fifties_per_innings: '50s per innings',
    bat_hundreds_per_innings: '100s per innings',
    bat_boundary_run_pct: 'Boundary runs %',
    bat_fours_per_innings: 'Fours per innings',
    bat_sixes_per_innings: 'Sixes per innings',
    bowl_wickets_per_innings: 'Wickets per bowling innings',
    bowl_average: 'Bowling average',
    bowl_economy: 'Economy rate',
    bowl_strike_rate: 'Bowling strike rate',
    bowl_dot_pct: 'Dot-ball %',
    bat_high_rrr_sr: 'Strike rate at required rate ≥12',
    bat_low_rrr_sr: 'Strike rate at required rate <6',
    bat_low_wickets_sr: 'Strike rate with ≤4 wickets in hand',
    bat_close_finish_runs_per_match: 'Runs per close-finish appearance',
    bat_last_over_sr: 'Final-over strike rate',
    bat_recovery_sr: 'Strike rate after early wickets',
    bowl_high_rrr_economy: 'Economy at required rate ≥12',
    bowl_low_rrr_economy: 'Economy at required rate <6',
    bowl_low_wickets_wickets_per_match: 'Wickets per low-wicket match',
    bowl_close_finish_economy: 'Economy in close finishes',
    bowl_last_over_economy: 'Final-over economy',
    bowl_low_total_defenses: 'Wins defending totals below 140',
    bat_winning_runs_per_match: 'Runs per appearance in wins',
    bat_17_run_final_overs: 'Final overs scoring 17+ runs (%)',
    bat_player_of_match: 'Recorded Player of the Match rate',
    bowl_winning_wickets_per_match: 'Wickets per appearance in wins',
    bowl_under_10_defenses: 'Successful sub-10 final-over defenses (%)',
    bowl_well_set_dismissals: 'Dismissals of batters on 30+ runs',
    bowl_partnership_breaks: 'Strong partnership break rate',
    bowl_player_of_match: 'Recorded Player of the Match rate',
    bat_last5_runs_per_innings: 'Last five appearances: runs per innings',
    bat_last10_runs_per_innings: 'Last ten appearances: runs per innings',
    bat_season_consistency: 'Multi-season batting consistency',
    bat_home_away_balance: 'Home/away batting balance',
    bat_strong_opposition_runs_per_match: 'Runs per match vs top-four teams',
    bat_positions_played: 'Batting positions used',
    bowl_last5_wickets_per_match: 'Last five appearances: wickets per match',
    bowl_last10_wickets_per_match: 'Last ten appearances: wickets per match',
    bowl_season_consistency: 'Multi-season bowling consistency',
    bowl_home_away_balance: 'Home/away bowling balance',
    bowl_strong_opposition_wickets_per_match: 'Wickets per match vs top-four teams',
    bowl_phase_roles_played: 'Bowling phases used',
    bat_powerplay_sr: 'Powerplay strike rate',
    bat_middle_overs_sr: 'Middle-over strike rate',
    bat_death_overs_sr: 'Death-over strike rate',
    bat_vs_pace_sr: 'Strike rate vs pace',
    bat_vs_spin_sr: 'Strike rate vs spin',
    bowl_powerplay_economy: 'Powerplay economy',
    bowl_middle_overs_economy: 'Middle-over economy',
    bowl_death_overs_economy: 'Death-over economy',
    bowl_vs_left_handed_economy: 'Economy vs left-handed batters',
    bowl_vs_right_handed_economy: 'Economy vs right-handed batters',
    bat_failure_rate: 'Innings below 10 runs (%)',
    bowl_expensive_spell_rate: 'Spells above 12 runs per over (%)',
    field_catches: 'Catches',
    field_runouts: 'Run-outs',
    field_stumpings: 'Stumpings',
    field_runs_saved: 'Runs saved',
    field_dropped_catches: 'Dropped catches'
  };

  let data;
  let profileChart;
  let masterChart;
  let situationChart;
  let battingChart;
  let bowlingChart;
  let comparisonChart;
  let activeTopTenRole = null;
  let selectedPlayerIds = [];

  const escapeHtml = value => String(value ?? '').replace(/[&<>"']/g, char => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
  }[char]));
  const display = (value, digits = 1) => value == null || value === '' || !Number.isFinite(Number(value)) ? 'Unavailable' : Number(value).toFixed(digits);
  const rateWeightMetric = key =>
    (key.startsWith('bat_') && (key.endsWith('_strike_rate') || key.endsWith('_sr'))) ||
    (key.startsWith('bowl_') && key.endsWith('_economy'));
  const displayRateMetric = (value, digits = 2) =>
    value == null || value === '' || !Number.isFinite(Number(value)) ? 'Data Not Available' : Number(value).toFixed(digits);
  const scoreText = value => value == null ? '—' : `${Number(value).toFixed(1)}`;

  function showError(message) {
    const target = document.getElementById('ranking-error');
    if (target) {
      target.hidden = false;
      target.textContent = message;
    }
  }

  function hideLegacyRankingPages() {
    ['overview', 'batters', 'bowlers', 'pressure', 'methodology'].forEach(id => {
      const page = document.getElementById(`page-${id}`);
      const nav = document.getElementById(`nav-${id}`);
      if (page) page.classList.remove('active');
      if (nav) nav.hidden = true;
    });
  }

  function showPage(id) {
    if (typeof window.showPage === 'function') window.showPage(id);
    document.getElementById('page-ipl-rankings')?.classList.toggle('active', id === 'ipl-rankings');
    document.getElementById('page-player-comparison')?.classList.toggle('active', id === 'player-comparison');
    document.getElementById('page-ranking-methodology')?.classList.toggle('active', id === 'ranking-methodology');
    document.getElementById('page-rr-auction-2027')?.classList.toggle('active', id === 'rr-auction-2027');
    document.getElementById('nav-ipl-rankings')?.classList.toggle('active', id === 'ipl-rankings');
    document.getElementById('nav-player-comparison')?.classList.toggle('active', id === 'player-comparison');
    document.getElementById('nav-ranking-methodology')?.classList.toggle('active', id === 'ranking-methodology');
    document.getElementById('nav-rr-auction-2027')?.classList.toggle('active', id === 'rr-auction-2027');
  }

  function renderMetadata() {
    const metadata = data.metadata;
    document.getElementById('ranking-filter-note').textContent =
      `Top 10 batters requires at least ${metadata.role_leaderboard_minimums.batting_runs} IPL runs; ` +
      `Top 10 bowlers requires at least ${metadata.role_leaderboard_minimums.bowling_wickets} IPL wickets; ` +
      'all-rounders are inferred from both available role scores.';
    document.getElementById('footer-ranking-scope').textContent =
      `${metadata.total_deliveries.toLocaleString()} IPL deliveries · ${metadata.total_matches} matches · ${metadata.seasons.join(', ')}`;
    document.getElementById('ranking-summary').innerHTML = `
      <article><span>Players ranked</span><strong>${metadata.players_ranked}</strong></article>
      <article><span>IPL matches</span><strong>${metadata.total_matches}</strong></article>
      <article><span>Ball-by-ball records</span><strong>${Number(metadata.total_deliveries).toLocaleString()}</strong></article>
      <article><span>Seasons</span><strong>${metadata.seasons.join(', ')}</strong></article>`;
    document.getElementById('ranking-weight-table').innerHTML = `
      <table class="ranking-table"><thead><tr><th>Category</th><th>Official weight</th></tr></thead>
      <tbody>${CATEGORY_KEYS.map(key => `<tr><td>${escapeHtml(CATEGORY_LABELS[key])}</td><td>${Number(data.weights[key]).toFixed(0)}%</td></tr>`).join('')}
      <tr><th>Total</th><th>${Number(metadata.weights_total_percent).toFixed(0)}%</th></tr></tbody></table>`;
    document.getElementById('ranking-definition-list').innerHTML = CATEGORY_KEYS.map(key => `
      <article class="ranking-leader-card">
        <span>${Number(data.weights[key]).toFixed(0)}% · ${escapeHtml(CATEGORY_LABELS[key])}</span>
        <small>${escapeHtml((data.categories[key]?.supported_metrics || []).map(metric => METRIC_LABELS[metric] || metric).join(' · ') || 'No supported metrics')}</small>
        ${(data.categories[key]?.unavailable_metrics || []).map(metric => `<small>Unavailable: ${escapeHtml(metric.key.replaceAll('_', ' '))}</small>`).join('')}
      </article>`).join('');
    const components = metadata.role_final_score_components_percent;
    const frameworkRows = [
      ['Batter', 'Six categories (90%) + overall strike rate (10%)', components.batter.total],
      ['Bowler', 'Six categories (90%) + overall economy (10%)', components.bowler.total],
      ['All-rounder', '50% batter final score + 50% bowler final score', components.all_rounder.total]
    ];
    document.getElementById('ranking-role-weight-table').innerHTML = `
      <h2>Role score frameworks</h2>
      <p>${escapeHtml(metadata.role_framework_policy)} The six category weights total 90%; the role-specific standalone rate contributes the remaining 10%. Missing contributions are not reweighted.</p>
      <table class="ranking-table"><thead><tr><th>Framework</th><th>Role distribution</th><th>Total</th></tr></thead>
      <tbody>${frameworkRows.map(([name, distribution, total]) =>
        `<tr><th>${escapeHtml(name)}</th><td>${escapeHtml(distribution)}</td><td>${Number(total).toFixed(0)}%</td></tr>`
      ).join('')}</tbody></table>`;
  }

  function renderCategoryLeaders() {
    const cards = CATEGORY_KEYS.map(key => {
      const player = data.players
        .filter(row => row.category_scores[key] != null)
        .sort((a, b) => b.category_scores[key] - a.category_scores[key])[0];
      return `<button class="ranking-leader-card" data-profile="${escapeHtml(player?.player_id ?? '')}">
        <span>${escapeHtml(CATEGORY_LABELS[key])} · ${Number(data.weights[key]).toFixed(0)}%</span>
        <strong>${escapeHtml(player?.name || 'Unavailable')}</strong>
        <small>${player ? `${player.category_scores[key].toFixed(1)} / 100` : 'No supported sample'}</small>
      </button>`;
    });
    document.getElementById('category-leaders').innerHTML = cards.join('');
    const metricLabels = {
      bat_strike_rate: { label: 'Batting strike rate', unit: 'runs / 100 balls', sample: 'balls faced' },
      bowl_economy: { label: 'Bowling economy', unit: 'runs / over', sample: 'legal balls' }
    };
    const metricCards = (data.top_metric_performers || []).map(metric => {
      const details = metricLabels[metric.key];
      if (!details) return '';
      return `<button class="ranking-leader-card" data-profile="${escapeHtml(metric.player_id)}">
        <span>${escapeHtml(details.label)} · 10% role component</span>
        <strong>${escapeHtml(metric.name)}</strong>
        <small>${Number(metric.value).toFixed(metric.key === 'bowl_economy' ? 2 : 1)} ${details.unit} · ${Number(metric.normalized_score).toFixed(1)} / 100</small>
        <small>${Number(metric.sample_size).toLocaleString()} ${details.sample} · separate from category scores</small>
      </button>`;
    });
    document.getElementById('metric-leaders').innerHTML = metricCards.join('');
    document.querySelectorAll('#category-leaders .ranking-leader-card, #metric-leaders .ranking-leader-card').forEach(button => {
      button.addEventListener('click', () => openProfile(button.dataset.profile));
    });
  }

  function renderLeaderboard(rows = data.players) {
    const leaderboardKey = {
      batting: 'top_batters',
      bowling: 'top_bowlers',
    }[activeTopTenRole];
    const qualifiedPlayerIds = leaderboardKey
      ? new Set(data[leaderboardKey].map(player => String(player.player_id)))
      : null;
    const eligible = rows.filter(player =>
      qualifiedPlayerIds === null || qualifiedPlayerIds.has(String(player.player_id))
    );
    const ranked = activeTopTenRole
      ? eligible.sort((a, b) =>
        b.role_scores[activeTopTenRole].final_score - a.role_scores[activeTopTenRole].final_score
      ).slice(0, 10)
      : eligible.slice(0, 10);
    const scoreLabel = activeTopTenRole === 'batting' ? 'Batter score'
      : activeTopTenRole === 'bowling' ? 'Bowler score'
        : 'Final score';
    const filterNote = activeTopTenRole === 'batting'
      ? 'Batter methodology: 90% from the six categories + 10% from normalized overall batting strike rate. Category weights are not reassigned for missing data.'
      : activeTopTenRole === 'bowling'
        ? 'Bowler methodology: 90% from the six categories + 10% from normalized overall bowling economy (lower is better). Category weights are not reassigned for missing data.'
        : `Top 10 batters requires at least ${data.metadata.role_leaderboard_minimums.batting_runs} IPL runs; Top 10 bowlers requires at least ${data.metadata.role_leaderboard_minimums.bowling_wickets} IPL wickets. Overall scores use the available role path or a 50/50 role blend.`;
    document.getElementById('ranking-filter-note').textContent = filterNote;
    document.getElementById('ranking-table-body').innerHTML = ranked.map((player, index) => {
      const scorePath = activeTopTenRole ? player.role_scores[activeTopTenRole] : null;
      const score = scorePath ? scorePath.final_score : player.final_score;
      const coverage = scorePath ? scorePath.data_coverage : player.data_coverage;
      const categoryScores = scorePath ? scorePath.category_scores : player.category_scores;
      return `
      <tr data-profile="${escapeHtml(player.player_id)}" tabindex="0">
        <td>${activeTopTenRole ? index + 1 : player.rank}</td>
        <td><strong>${escapeHtml(player.name)}</strong><small>${escapeHtml(player.overall_role.replaceAll('_', ' '))} scoring path · ${player.sample.matches} matches${player.limited_sample ? ' · limited sample' : ''}</small></td>
        <td><strong>${Number(score).toFixed(2)}</strong><small>${Number(coverage).toFixed(0)}% data coverage</small></td>
        <td class="batting-strike-rate-column"><strong>${displayRateMetric(player.batting.strike_rate)}</strong><small>${player.sample.balls_faced.toLocaleString()} balls faced</small></td>
        <td class="bowling-economy-column"><strong>${displayRateMetric(player.bowling.economy)}</strong><small>${player.sample.legal_balls_bowled.toLocaleString()} legal balls</small></td>
        ${CATEGORY_KEYS.map(key => `<td>${scoreText(categoryScores[key])}</td>`).join('')}
      </tr>`;
    }).join('') || '<tr><td colspan="11">No players match these filters.</td></tr>';
    const header = document.querySelector('.ranking-table[aria-label="Official IPL player leaderboard"] thead tr th:nth-child(3)');
    if (header) header.textContent = scoreLabel;
    const battingRateHeader = document.querySelector('.ranking-table[aria-label="Official IPL player leaderboard"] thead .batting-strike-rate-column');
    const bowlingRateHeader = document.querySelector('.ranking-table[aria-label="Official IPL player leaderboard"] thead .bowling-economy-column');
    if (battingRateHeader) {
      battingRateHeader.textContent = activeTopTenRole === 'batting' ? 'Batting SR (10%)' : 'Batting SR';
    }
    if (bowlingRateHeader) {
      bowlingRateHeader.textContent = activeTopTenRole === 'bowling' ? 'Bowling economy (10%)' : 'Bowling economy';
    }
    document.querySelectorAll('.batting-strike-rate-column').forEach(cell => {
      cell.style.display = activeTopTenRole === 'bowling' ? 'none' : '';
    });
    document.querySelectorAll('.bowling-economy-column').forEach(cell => {
      cell.style.display = activeTopTenRole === 'batting' ? 'none' : '';
    });
    document.querySelectorAll('#ranking-table-body tr[data-profile]').forEach(row => {
      row.addEventListener('click', () => openProfile(row.dataset.profile));
      row.addEventListener('keydown', event => {
        if (event.key === 'Enter' || event.key === ' ') openProfile(row.dataset.profile);
      });
    });
    ['batting', 'bowling'].forEach(topTenRole => {
      const buttonId = topTenRole === 'batting' ? 'ranking-top-batters' : 'ranking-top-bowlers';
      const button = document.getElementById(buttonId);
      button.classList.toggle('is-active', activeTopTenRole === topTenRole);
      button.setAttribute('aria-pressed', String(activeTopTenRole === topTenRole));
    });
  }

  function metricMarkup(player) {
    const rows = Object.entries(player.metrics).map(([key, metric]) => {
      const label = METRIC_LABELS[key] || key.replaceAll('_', ' ');
      const value = metric.available ? display(metric.value, 2) : (rateWeightMetric(key) ? 'Data Not Available' : 'Unavailable');
      const sample = metric.sample_size == null ? '—' : metric.sample_size;
      const note = metric.available ? `${metric.source}; sample ${sample}` : (metric.reason || `No qualifying sample; sample ${sample}`);
      return `<tr><td>${escapeHtml(label)}</td><td>${escapeHtml(value)}</td><td>${escapeHtml(note)}</td></tr>`;
    });
    return rows.join('');
  }

  function openProfile(playerId) {
    const player = data.players.find(row => String(row.player_id) === String(playerId));
    if (!player) return;
    const target = document.getElementById('ranking-profile');
    target.hidden = false;
    target.innerHTML = `
      <div class="ranking-profile-heading">
        <div><p class="ranking-eyebrow">Official player profile · rank #${player.rank}</p>
        <h2>${escapeHtml(player.name)}</h2>
        <p>${player.overall_role.replaceAll('_', ' ')} final score ${player.final_score.toFixed(2)} / 100 · ${player.data_coverage.toFixed(1)}% weighted data coverage
          ${player.limited_sample ? ' · Limited sample' : ''}</p></div>
        <button type="button" class="ranking-button" id="close-profile">Close profile</button>
      </div>
      <div class="ranking-stats-grid">
        <article><span>Batter score · 10% SR component</span><strong>${scoreText(player.role_scores.batting.final_score)}</strong></article>
        <article><span>Bowler score · 10% economy component</span><strong>${scoreText(player.role_scores.bowling.final_score)}</strong></article>
        <article><span>All-rounder score · 50/50 blend</span><strong>${scoreText(player.role_scores.all_rounder?.final_score)}</strong></article>
        <article><span>Overall ranking role</span><strong>${escapeHtml(player.overall_role.replaceAll('_', ' '))}</strong></article>
      </div>
      <div class="ranking-stats-grid">
        <article><span>Matches</span><strong>${player.sample.matches}</strong></article>
        <article><span>Batting</span><strong>${player.batting.runs == null ? 'Unavailable' : `${player.batting.runs} runs`}</strong><small>${player.sample.balls_faced} balls faced</small></article>
        <article><span>Bowling</span><strong>${player.bowling.wickets == null ? 'Unavailable' : `${player.bowling.wickets} wickets`}</strong><small>${player.sample.legal_balls_bowled} legal balls</small></article>
        <article><span>Sample status</span><strong>${player.limited_sample ? 'Limited' : 'Meets threshold'}</strong><small>Minimum ${player.minimum_sample_matches} matches</small></article>
      </div>
      <div class="ranking-chart-grid">
        <article><h3>Six-category breakdown</h3><canvas id="profile-category-chart"></canvas></article>
        <article><h3>Master performance scores</h3><canvas id="profile-master-chart"></canvas><p>These indicators are not added to the final score.</p></article>
        <article><h3>Match-situation performance</h3><canvas id="profile-situation-chart"></canvas><p>Displayed values are derived from supported ball-by-ball samples.</p></article>
        <article><h3>Batting statistics</h3><canvas id="profile-batting-chart"></canvas></article>
        <article><h3>Bowling statistics</h3><canvas id="profile-bowling-chart"></canvas></article>
      </div>
      <div class="ranking-stats-grid">
        <article><span>Batting average</span><strong>${display(player.batting.average, 2)}</strong></article>
        <article><span>Batting strike rate</span><strong>${displayRateMetric(player.batting.strike_rate)}</strong></article>
        <article><span>Fours / sixes</span><strong>${player.batting.fours ?? 'Unavailable'} / ${player.batting.sixes ?? 'Unavailable'}</strong></article>
        <article><span>Fifties / hundreds</span><strong>${player.batting.fifties} / ${player.batting.hundreds}</strong></article>
        <article><span>Boundary runs</span><strong>${display(player.batting.boundary_run_percent, 2)}%</strong></article>
        <article><span>Bowling average</span><strong>${display(player.bowling.average, 2)}</strong></article>
        <article><span>Economy</span><strong>${displayRateMetric(player.bowling.economy)}</strong></article>
        <article><span>Bowling strike rate</span><strong>${display(player.bowling.strike_rate, 2)}</strong></article>
        <article><span>Wickets per innings</span><strong>${display(player.bowling.wickets_per_innings, 2)}</strong></article>
        <article><span>Dot balls</span><strong>${display(player.bowling.dot_ball_percent, 2)}%</strong></article>
      </div>
      <h3>Player-wise metric evidence</h3>
      <div class="ranking-table-scroll"><table class="ranking-table">
        <thead><tr><th>Metric</th><th>Observed value</th><th>Coverage / sample note</th></tr></thead>
        <tbody>${metricMarkup(player)}</tbody>
      </table></div>`;
    target.scrollIntoView({ behavior: 'smooth', block: 'start' });
    document.getElementById('close-profile').addEventListener('click', () => { target.hidden = true; });
    drawProfileCharts(player);
  }

  function chart(id, previous, type, labels, datasets, options = {}) {
    const canvas = document.getElementById(id);
    if (!canvas || typeof Chart === 'undefined') return previous;
    if (previous) previous.destroy();
    return new Chart(canvas, {
      type,
      data: { labels, datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: type === 'radar' ? { r: { min: 0, max: 100, ticks: { stepSize: 20 } } } : { y: { min: 0, max: 100 } },
        plugins: { legend: { display: datasets.length > 1 } },
        ...options
      }
    });
  }

  function drawProfileCharts(player) {
    const rolePaths = [
      ['Batter', player.role_scores.batting, '#EA1B87'],
      ['Bowler', player.role_scores.bowling, '#004BA0'],
      ['All-rounder', player.role_scores.all_rounder, '#176B45']
    ].filter(([, path]) => path?.final_score != null);
    profileChart = chart('profile-category-chart', profileChart, 'radar',
      CATEGORY_KEYS.map(key => CATEGORY_LABELS[key]),
      rolePaths.map(([label, path, color]) => ({
        label: `${label} category scores`,
        data: CATEGORY_KEYS.map(key => path.category_scores[key]),
        backgroundColor: 'transparent',
        borderColor: color,
        pointBackgroundColor: color
      })));

    const masterKeys = Object.keys(MASTER_LABELS);
    masterChart = chart('profile-master-chart', masterChart, 'bar',
      masterKeys.map(key => MASTER_LABELS[key]),
      [{ label: 'Master performance score', data: masterKeys.map(key => player.master_scores[key]), backgroundColor: ['#EA1B87', '#004BA0', '#FFB81C'] }]);

    const situationKeys = ['bat_high_rrr_sr', 'bat_low_rrr_sr', 'bat_low_wickets_sr', 'bat_last_over_sr', 'bowl_high_rrr_economy', 'bowl_last_over_economy'];
    const situationLabels = ['RRR ≥12 SR', 'RRR <6 SR', '≤4 wickets SR', 'Last-over SR', 'RRR ≥12 econ', 'Last-over econ'];
    const situationMetrics = situationKeys.map(key => player.metrics[key]);
    situationChart = chart('profile-situation-chart', situationChart, 'bar', situationLabels, [{
      label: 'Observed metric (not normalized)',
      data: situationMetrics.map(metric => metric?.available ? metric.value : null),
      backgroundColor: '#176B45'
    }], { scales: { y: { beginAtZero: true } } });

    battingChart = chart('profile-batting-chart', battingChart, 'bar',
      ['Runs', 'Average', 'Strike rate', 'Fifties', 'Hundreds', 'Fours', 'Sixes'],
      [{ label: 'Observed batting statistics', data: [
        player.batting.runs, player.batting.average, player.batting.strike_rate,
        player.batting.fifties, player.batting.hundreds, player.batting.fours, player.batting.sixes
      ], backgroundColor: '#EA1B87' }],
      { scales: { y: { beginAtZero: true } } });
    bowlingChart = chart('profile-bowling-chart', bowlingChart, 'bar',
      ['Wickets', 'Average', 'Economy', 'Strike rate', 'Wickets / innings', 'Dot-ball %'],
      [{ label: 'Observed bowling statistics', data: [
        player.bowling.wickets, player.bowling.average, player.bowling.economy, player.bowling.strike_rate,
        player.bowling.wickets_per_innings, player.bowling.dot_ball_percent
      ], backgroundColor: '#004BA0' }],
      { scales: { y: { beginAtZero: true } } });
  }

  function selectedPlayers() {
    return selectedPlayerIds
      .map(playerId => data.players.find(player => String(player.player_id) === playerId))
      .filter(Boolean);
  }

  function renderSelectedPlayers() {
    const players = selectedPlayers();
    document.getElementById('compare-selection-count').textContent =
      `${players.length} player${players.length === 1 ? '' : 's'} selected`;
    document.getElementById('compare-selection-message').textContent =
      players.length < 2 ? 'Select at least two players to compare.' : `Comparing ${players.length} players.`;
    document.getElementById('compare-selected-players').innerHTML = players.map(player => `
      <span class="compare-player-chip">
        ${escapeHtml(player.name)}
        <button type="button" data-remove-player="${escapeHtml(player.player_id)}"
          aria-label="Remove ${escapeHtml(player.name)}">×</button>
      </span>`).join('') || '<span class="compare-empty">No players selected.</span>';
  }

  function renderCompareSearchResults() {
    const input = document.getElementById('compare-player-search');
    const results = document.getElementById('compare-search-results');
    const query = input.value.trim().toLocaleLowerCase();
    if (!query) {
      results.hidden = true;
      results.innerHTML = '';
      input.setAttribute('aria-expanded', 'false');
      return;
    }

    const selected = new Set(selectedPlayerIds);
    const matches = data.players
      .filter(player => !selected.has(String(player.player_id)) && player.name.toLocaleLowerCase().includes(query))
      .slice(0, 8);
    results.innerHTML = matches.length
      ? matches.map(player => `<button type="button" role="option" aria-selected="false"
          data-add-player="${escapeHtml(player.player_id)}">${escapeHtml(player.name)} · #${player.rank}</button>`).join('')
      : '<div class="compare-empty">No matching players found.</div>';
    results.hidden = false;
    input.setAttribute('aria-expanded', 'true');
  }

  function addComparisonPlayer(playerId) {
    const id = String(playerId);
    if (!selectedPlayerIds.includes(id)) selectedPlayerIds.push(id);
    document.getElementById('compare-player-search').value = '';
    renderSelectedPlayers();
    renderCompareSearchResults();
    renderComparison();
  }

  function renderComparison() {
    const players = selectedPlayers();
    const target = document.getElementById('comparison-results');
    if (players.length < 2) {
      target.innerHTML = '<p>Add at least two players using the search above.</p>';
      if (comparisonChart) comparisonChart.destroy();
      return;
    }
    const rows = [
      ['Final score', player => player.final_score, true],
      ['Batter final score', player => player.role_scores.batting.final_score, true],
      ['Bowler final score', player => player.role_scores.bowling.final_score, true],
      ['All-rounder final score', player => player.role_scores.all_rounder?.final_score, true],
      ...CATEGORY_KEYS.map(key => [CATEGORY_LABELS[key], player => player.category_scores[key], true]),
      ...Object.entries(MASTER_LABELS).map(([key, label]) => [label, player => player.master_scores[key], true]),
      ['Data coverage', player => player.data_coverage, false],
      ['Matches', player => player.sample.matches, false],
      ['Batting runs', player => player.batting.runs, false],
      ['Batting average', player => player.batting.average, false],
      ['Batting strike rate', player => displayRateMetric(player.batting.strike_rate), false],
      ['Fours / sixes', player => player.batting.fours == null || player.batting.sixes == null ? null : `${player.batting.fours} / ${player.batting.sixes}`, false],
      ['Bowling wickets', player => player.bowling.wickets, false],
      ['Bowling average', player => player.bowling.average, false],
      ['Bowling economy', player => displayRateMetric(player.bowling.economy), false],
      ['Bowling strike rate', player => player.bowling.strike_rate, false]
    ];
    target.innerHTML = `<div class="ranking-table-scroll"><table class="ranking-table">
      <thead><tr><th>Metric</th>${players.map(player => `<th>${escapeHtml(player.name)}</th>`).join('')}</tr></thead>
      <tbody>${rows.map(([label, getter, highlight]) => {
        const values = players.map(getter);
        const max = Math.max(...values.filter(Number.isFinite));
        return `<tr><th>${escapeHtml(label)}</th>${values.map(value =>
          `<td class="${highlight && Number.isFinite(value) && value === max ? 'ranking-better' : ''}">${typeof value === 'string' ? escapeHtml(value) : display(value, 2)}</td>`
        ).join('')}</tr>`;
      }).join('')}</tbody></table></div>`;
    comparisonChart = chart('comparison-chart', comparisonChart, 'radar',
      ['Final', ...CATEGORY_KEYS.map(key => CATEGORY_LABELS[key])],
      players.map(player => ({
        label: player.name,
        data: [player.final_score, ...CATEGORY_KEYS.map(key => player.category_scores[key])],
        borderColor: ['#EA1B87', '#004BA0', '#FFB81C', '#176B45'][players.indexOf(player) % 4],
        backgroundColor: 'transparent'
      })));
  }

  function populateComparison() {
    selectedPlayerIds = data.players.slice(0, 2).map(player => String(player.player_id));
    renderSelectedPlayers();
    renderComparison();
  }

  async function load() {
    hideLegacyRankingPages();
    showPage('ipl-rankings');
    try {
      const response = await fetch('/dashboard_data.json', { cache: 'no-store' });
      if (!response.ok) throw new Error(`Ranking data request failed (${response.status}).`);
      data = await response.json();
      if (!Array.isArray(data.players) || !data.players.length) throw new Error('Ranking data contains no player leaderboard.');
      renderMetadata();
      renderCategoryLeaders();
      renderLeaderboard();
      populateComparison();
      document.getElementById('ranking-top-batters').addEventListener('click', () => {
        activeTopTenRole = activeTopTenRole === 'batting' ? null : 'batting';
        renderLeaderboard();
      });
      document.getElementById('ranking-top-bowlers').addEventListener('click', () => {
        activeTopTenRole = activeTopTenRole === 'bowling' ? null : 'bowling';
        renderLeaderboard();
      });
      const compareSearch = document.getElementById('compare-player-search');
      const compareResults = document.getElementById('compare-search-results');
      compareSearch.addEventListener('input', renderCompareSearchResults);
      compareSearch.addEventListener('keydown', event => {
        if (event.key === 'Escape') {
          compareSearch.value = '';
          renderCompareSearchResults();
        }
        if (event.key === 'Enter') {
          const firstResult = compareResults.querySelector('[data-add-player]');
          if (firstResult) {
            event.preventDefault();
            addComparisonPlayer(firstResult.dataset.addPlayer);
          }
        }
      });
      compareResults.addEventListener('click', event => {
        const button = event.target.closest('[data-add-player]');
        if (button) addComparisonPlayer(button.dataset.addPlayer);
      });
      document.getElementById('compare-selected-players').addEventListener('click', event => {
        const button = event.target.closest('[data-remove-player]');
        if (!button) return;
        selectedPlayerIds = selectedPlayerIds.filter(playerId => playerId !== button.dataset.removePlayer);
        renderSelectedPlayers();
        renderCompareSearchResults();
        renderComparison();
      });
      document.getElementById('compare-clear-players').addEventListener('click', () => {
        selectedPlayerIds = [];
        renderSelectedPlayers();
        renderComparison();
      });
      document.getElementById('nav-ipl-rankings').addEventListener('click', () => showPage('ipl-rankings'));
      document.getElementById('nav-player-comparison').addEventListener('click', () => showPage('player-comparison'));
      document.getElementById('nav-ranking-methodology').addEventListener('click', () => showPage('ranking-methodology'));
      document.getElementById('ranking-compare-button').addEventListener('click', () => showPage('player-comparison'));
    } catch (error) {
      showError(`${error.message} Check that the dashboard data file is deployed.`);
    }
  }

  document.addEventListener('DOMContentLoaded', load, { once: true });
})();
