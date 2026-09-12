(() => {
  const article = document.querySelector('#articleBody');
  const progress = document.querySelector('.reading-progress span');
  const topButton = document.querySelector('.back-to-top');
  const menuButton = document.querySelector('.menu-button');
  const mobileMenu = document.querySelector('#mobile-menu');

  const updateScroll = () => {
    const max = document.documentElement.scrollHeight - window.innerHeight;
    const ratio = max > 0 ? window.scrollY / max : 0;
    if (progress) progress.style.width = `${Math.min(100, ratio * 100)}%`;
    if (topButton) topButton.classList.toggle('visible', window.scrollY > 700);
  };
  window.addEventListener('scroll', updateScroll, { passive: true });
  updateScroll();
  topButton?.addEventListener('click', () => window.scrollTo({ top: 0, behavior: 'smooth' }));

  menuButton?.addEventListener('click', () => {
    const open = menuButton.getAttribute('aria-expanded') === 'true';
    menuButton.setAttribute('aria-expanded', String(!open));
    mobileMenu.hidden = open;
  });

  if (!article) return;

  const words = article.textContent.replace(/\s+/g, '').length;
  const readingTime = document.querySelector('#readingTime');
  if (readingTime) readingTime.textContent = `${Math.max(1, Math.ceil(words / 420))} 分钟`;

  document.querySelectorAll('a[href="02-experiment-log.md"]').forEach(a => a.href = '02-experiment-log.html');
  document.querySelectorAll('a[href="01-beginner-guide.md"]').forEach(a => a.href = '01-beginner-guide.html');

  const headings = [...article.querySelectorAll('h2, h3')];
  const slug = (text, index) => text.trim().toLowerCase()
    .replace(/[\s/：:，,。.？?！!（）()]+/g, '-')
    .replace(/[^\w\u3400-\u9fff-]/g, '')
    .replace(/^-+|-+$/g, '') || `section-${index + 1}`;
  headings.forEach((heading, index) => {
    if (!heading.id) heading.id = slug(heading.textContent, index);
  });

  const makeToc = (target) => {
    if (!target) return;
    headings.forEach(heading => {
      const link = document.createElement('a');
      link.href = `#${heading.id}`;
      link.textContent = heading.textContent;
      link.className = `level-${heading.tagName.slice(1)}`;
      link.dataset.target = heading.id;
      link.addEventListener('click', () => {
        if (mobileMenu) mobileMenu.hidden = true;
        menuButton?.setAttribute('aria-expanded', 'false');
      });
      target.append(link);
    });
  };
  makeToc(document.querySelector('#tocList'));
  makeToc(document.querySelector('#mobileTocList'));

  if ('IntersectionObserver' in window) {
    const observer = new IntersectionObserver(entries => {
      const visible = entries.filter(entry => entry.isIntersecting).sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top)[0];
      if (!visible) return;
      document.querySelectorAll('.toc a').forEach(link => link.classList.toggle('active', link.dataset.target === visible.target.id));
    }, { rootMargin: '-90px 0px -72% 0px', threshold: 0 });
    headings.forEach(heading => observer.observe(heading));
  }

  article.querySelectorAll('table').forEach(table => {
    if (table.parentElement?.classList.contains('table-wrap')) return;
    const wrapper = document.createElement('div');
    wrapper.className = 'table-wrap';
    table.parentNode.insertBefore(wrapper, table);
    wrapper.append(table);
  });

  article.querySelectorAll('pre').forEach(pre => {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'copy-code';
    button.textContent = '复制';
    button.addEventListener('click', async () => {
      try {
        await navigator.clipboard.writeText(pre.querySelector('code')?.textContent || pre.textContent);
        button.textContent = '已复制';
      } catch (_) {
        button.textContent = '复制失败';
      }
      window.setTimeout(() => button.textContent = '复制', 1400);
    });
    pre.append(button);
  });

  const lightbox = document.querySelector('.lightbox');
  const lightboxImage = lightbox?.querySelector('img');
  const closeLightbox = () => { if (lightbox) lightbox.hidden = true; };
  article.querySelectorAll('img').forEach(img => img.addEventListener('click', () => {
    if (!lightbox || !lightboxImage) return;
    lightboxImage.src = img.currentSrc || img.src;
    lightboxImage.alt = img.alt;
    lightbox.hidden = false;
  }));
  lightbox?.querySelector('button')?.addEventListener('click', closeLightbox);
  lightbox?.addEventListener('click', event => { if (event.target === lightbox) closeLightbox(); });
  document.addEventListener('keydown', event => { if (event.key === 'Escape') closeLightbox(); });
})();
