(() => {
  let presentationData;
  let presentationRequest;
  let lastSelectedPlayer;

  const escapeHtml = value => String(value ?? '').replace(/[&<>"']/g, character => ({
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#39;'
  }[character]));

  const renderPlayerList = role => {
    const players = presentationData.target_lists[role];
    const list = document.getElementById('auction-player-list');
    const profile = document.getElementById('auction-player-profile');
    const roleLabel = role === 'batting' ? 'Batters' : 'Bowlers';

    profile.hidden = true;
    list.innerHTML = players.map(player => `
      <button class="auction-player-card" type="button" data-auction-player-role="${role}" data-auction-player-rank="${player.rank}">
        <span class="auction-player-rank">#${player.rank} · ${escapeHtml(player.country)}</span>
        <strong>${escapeHtml(player.name)}</strong>
        <small>${escapeHtml(player.role)} · ${escapeHtml(player.reason)}</small>
      </button>`).join('');
    document.getElementById('auction-target-status').textContent =
      `Top 10 ${roleLabel} from the presentation. Availability and auction status are shown as provided in the source deck.`;
    ['batting', 'bowling'].forEach(key => {
      const button = document.getElementById(key === 'batting' ? 'auction-top-batters' : 'auction-top-bowlers');
      button.classList.toggle('is-active', key === role);
      button.setAttribute('aria-pressed', String(key === role));
    });
  };

  const labeledFields = (lines, labels) => {
    const fields = [];
    for (let index = 0; index < lines.length - 1; index += 1) {
      if (!labels.includes(lines[index])) continue;
      fields.push({ label: lines[index], value: lines[index + 1] });
      index += 1;
    }
    return fields;
  };

  const profileSections = player => {
    const analysisIndex = player.profile.findIndex(line => line.includes('SCOUTING ANALYSIS'));
    const overview = player.profile.slice(0, analysisIndex);
    const analysis = player.profile.slice(analysisIndex + 1);
    const statisticsIndex = overview.indexOf('STATISTICS');
    const seasonIndex = overview.indexOf('BEST IPL SEASON', statisticsIndex);
    const metricIndex = overview.indexOf('Metric', statisticsIndex);
    const statistics = [];

    if (metricIndex !== -1 && seasonIndex !== -1) {
      for (let index = metricIndex + 3; index + 2 < seasonIndex; index += 3) {
        statistics.push({
          metric: overview[index],
          t20: overview[index + 1],
          ipl: overview[index + 2]
        });
      }
    }

    const contractIndex = overview.findIndex(line =>
      line.startsWith('CONTRACT STATUS') || line.startsWith('UNSOLD IN IPL')
    );
    const verdictIndex = overview.indexOf('FINAL VERDICT');
    const nextSectionIndex = values => {
      const positions = values.map(value => overview.findIndex(line => line.startsWith(value)))
        .filter(index => index > -1);
      return positions.length ? Math.min(...positions) : verdictIndex;
    };
    const whyIndex = overview.indexOf('WHY RR SHOULD SELECT HIM');
    const noteEnd = nextSectionIndex(['CONTRACT STATUS', 'UNSOLD IN IPL 2026 AUCTION', 'FINAL VERDICT']);
    const contractEnd = verdictIndex > -1 ? verdictIndex : overview.length;
    const season = seasonIndex > -1 ? overview[seasonIndex + 1] : '';
    const analysisMarkers = [
      ['WHY RR SHOULD SELECT HIM', 'A  WHY SHOULD RR SELECT HIM?'],
      ['STRENGTH HE ADDS', 'B  STRENGTH HE ADDS'],
      ['RR PROBLEM HE SOLVES', 'C  RR PROBLEM HE SOLVES']
    ];
    const analysisBlocks = analysisMarkers.map(([title, marker], blockIndex) => {
      const start = analysis.findIndex(line => line.startsWith(marker));
      const followingMarkers = analysisMarkers.slice(blockIndex + 1).map(([, next]) => next);
      followingMarkers.push('PLAYER IMPACT SUMMARY', 'FINAL VERDICT');
      const endPositions = followingMarkers
        .map(next => analysis.findIndex(line => line.startsWith(next)))
        .filter(index => index > start);
      const end = endPositions.length ? Math.min(...endPositions) : analysis.length;
      return { title, content: start > -1 ? analysis.slice(start + 1, end) : [] };
    });
    const impactIndex = analysis.indexOf('PLAYER IMPACT SUMMARY');
    const impactEnd = analysis.findIndex((line, index) => index > impactIndex && line === 'FINAL VERDICT');
    const impactSummary = labeledFields(
      impactIndex > -1 ? analysis.slice(impactIndex + 1, impactEnd > -1 ? impactEnd : undefined) : [],
      ['BEST RR ROLE', 'PRIMARY STRENGTH', 'RR PROBLEM SOLVED', 'MAIN CONCERN',
        'SELECTION PRIORITY', 'SQUAD FIT', 'ACQUISITION FEASIBILITY', 'FINAL VERDICT']
    );
    const analysisVerdictIndex = analysis.findIndex(line => line === 'FINAL VERDICT');
    const verdictLogic = analysis.find(line => line.startsWith('VERDICT LOGIC'));

    return {
      facts: overview.filter(line => /^(Country|Role|Bat|Bowl|Age):/.test(line)),
      statistics,
      season,
      scoutingNotes: whyIndex > -1
        ? overview.slice(whyIndex + 1, noteEnd > whyIndex ? noteEnd : verdictIndex)
        : [],
      auctionFields: labeledFields(
        contractIndex > -1 ? overview.slice(contractIndex + 1, contractEnd) : [],
        ['Auction status', 'Franchise', 'Contract', 'Route', 'RR feasibility']
      ),
      verdict: verdictIndex > -1 ? overview.slice(verdictIndex + 1, analysisIndex) : [],
      analysisBlocks,
      impactSummary,
      analysisScore: analysisVerdictIndex > -1 ? analysis[analysisVerdictIndex + 1] : '',
      verdictLogic: verdictLogic || ''
    };
  };

  const showPlayerProfile = player => {
    const sections = profileSections(player);
    const profile = document.getElementById('auction-player-profile');
    document.getElementById('auction-player-profile-name').textContent = player.name;
    document.getElementById('auction-player-profile-meta').textContent =
      `${player.country} · ${player.role} · ${player.auction_status}`;
    const factCards = [
      { label: 'Shortlist role', value: player.role },
      { label: 'Country', value: player.country },
      { label: 'Auction status in source deck', value: player.auction_status },
      { label: 'Franchise in source deck', value: player.franchise }
    ];
    const statisticsTable = sections.statistics.length ? `
      <div class="auction-profile-table-wrap">
        <table class="auction-profile-table">
          <caption>T20 and IPL statistics shown in the source profile</caption>
          <thead><tr><th>Metric</th><th>T20 (2022–26)</th><th>IPL</th></tr></thead>
          <tbody>${sections.statistics.map(stat => `
            <tr><th scope="row">${escapeHtml(stat.metric)}</th><td>${escapeHtml(stat.t20)}</td><td>${escapeHtml(stat.ipl)}</td></tr>`
          ).join('')}</tbody>
        </table>
      </div>` : '<p>Statistics are unavailable in this player profile.</p>';
    const fieldCards = fields => fields.map(field => `
      <div class="auction-profile-fact"><span>${escapeHtml(field.label)}</span><strong>${escapeHtml(field.value)}</strong></div>`
    ).join('');
    const noteList = sections.scoutingNotes.filter(line =>
      !line.startsWith('Source:') && !line.startsWith('RAJASTHAN ROYALS')
    );
    const analysisMarkup = sections.analysisBlocks.map(block => `
      <section class="auction-profile-card">
        <h4>${escapeHtml(block.title)}</h4>
        ${block.content.map(line => `<p>${escapeHtml(line)}</p>`).join('')}
      </section>`).join('');
    const impactMarkup = sections.impactSummary.length ? `
      <section class="auction-profile-card">
        <h4>Player impact summary</h4>
        <div class="auction-profile-facts">${fieldCards(sections.impactSummary)}</div>
      </section>` : '';
    const auctionFields = sections.auctionFields.length ? sections.auctionFields : [
      { label: 'Auction status', value: player.auction_status },
      { label: 'Franchise', value: player.franchise }
    ];

    document.getElementById('auction-player-profile-content').innerHTML = `
      <div class="auction-profile-facts">${fieldCards(factCards)}</div>
      <section class="auction-profile-card">
        <h4>Performance statistics</h4>
        ${statisticsTable}
      </section>
      ${sections.season ? `<section class="auction-profile-card"><h4>Best IPL season</h4><p>${escapeHtml(sections.season)}</p></section>` : ''}
      <section class="auction-profile-card">
        <h4>Why RR should consider this player</h4>
        <p>${escapeHtml(player.reason)}</p>
        ${noteList.map(line => `<p>${escapeHtml(line)}</p>`).join('')}
      </section>
      <section class="auction-profile-card">
        <h4>Auction and squad fit</h4>
        <div class="auction-profile-facts">${fieldCards(auctionFields)}</div>
      </section>
      <div class="auction-profile-analysis">${analysisMarkup}</div>
      ${impactMarkup}
      ${sections.analysisScore ? `<p class="auction-profile-score"><strong>Final verdict:</strong> ${escapeHtml(sections.analysisScore)}</p>` : ''}
      ${sections.verdict.length ? `<p class="auction-profile-verdict">${sections.verdict.map(escapeHtml).join(' · ')}</p>` : ''}
      ${sections.verdictLogic ? `<section class="auction-profile-card"><h4>Scouting verdict</h4><p>${escapeHtml(sections.verdictLogic)}</p></section>` : ''}`;
    profile.hidden = false;
    profile.scrollIntoView({ behavior: 'smooth', block: 'start' });
  };

  const loadAuctionDetails = () => {
    if (presentationData) return Promise.resolve(presentationData);
    if (presentationRequest) return presentationRequest;

    const status = document.getElementById('auction-details-status');
    const report = document.getElementById('auction-presentation-content');
    report.dataset.loading = 'true';
    presentationRequest = fetch('/rr_auction_2027_details.json', { cache: 'no-store' })
      .then(response => {
        if (!response.ok) throw new Error(`Presentation details request failed (${response.status}).`);
        return response.json();
      })
      .then(presentation => {
        if (!Array.isArray(presentation.content) || !presentation.content.length ||
            !Array.isArray(presentation.target_lists?.batting) ||
            !Array.isArray(presentation.target_lists?.bowling) ||
            presentation.target_lists.batting.length !== 10 ||
            presentation.target_lists.bowling.length !== 10) {
          throw new Error('Presentation contents or player shortlists are incomplete.');
        }
        presentationData = presentation;
        report.innerHTML = presentation.content.map(line => `<p>${escapeHtml(line)}</p>`).join('');
        report.dataset.loaded = 'true';
        status.textContent = `All ${presentation.content_count} text items from the presentation are shown below. Original charts, images, and slide formatting are preserved in the downloadable PowerPoint.`;
        status.setAttribute('role', 'status');
        return presentation;
      })
      .catch(error => {
        status.textContent = `Unable to load presentation details: ${error.message}`;
        status.setAttribute('role', 'alert');
        return null;
      })
      .finally(() => {
        delete report.dataset.loading;
        presentationRequest = null;
      });
    return presentationRequest;
  };

  document.addEventListener('DOMContentLoaded', () => {
    document.getElementById('nav-rr-auction-2027')
      .addEventListener('click', loadAuctionDetails);

    document.getElementById('auction-top-batters').addEventListener('click', async () => {
      if (await loadAuctionDetails()) renderPlayerList('batting');
    });
    document.getElementById('auction-top-bowlers').addEventListener('click', async () => {
      if (await loadAuctionDetails()) renderPlayerList('bowling');
    });
    document.getElementById('auction-player-list').addEventListener('click', event => {
      const button = event.target.closest('[data-auction-player-role]');
      if (!button || !presentationData) return;
      const role = button.dataset.auctionPlayerRole;
      const rank = Number(button.dataset.auctionPlayerRank);
      const player = presentationData.target_lists[role].find(item => item.rank === rank);
      if (!player) return;
      lastSelectedPlayer = button;
      showPlayerProfile(player);
    });
    document.getElementById('auction-close-player-profile').addEventListener('click', () => {
      document.getElementById('auction-player-profile').hidden = true;
      lastSelectedPlayer?.focus();
    });

    document.getElementById('auction-content-search').addEventListener('input', event => {
      const query = event.target.value.trim().toLocaleLowerCase();
      const entries = [...document.querySelectorAll('#auction-presentation-content p')];
      let visibleEntries = 0;
      entries.forEach(entry => {
        entry.hidden = !entry.textContent.toLocaleLowerCase().includes(query);
        if (!entry.hidden) visibleEntries += 1;
      });
      if (document.getElementById('auction-presentation-content').dataset.loaded === 'true') {
        document.getElementById('auction-details-status').textContent = query
          ? `${visibleEntries} of ${entries.length} content items match “${event.target.value.trim()}”.`
          : `All ${entries.length} text items from the presentation are shown below. Original charts, images, and slide formatting are preserved in the downloadable PowerPoint.`;
      }
    });
  }, { once: true });
})();
