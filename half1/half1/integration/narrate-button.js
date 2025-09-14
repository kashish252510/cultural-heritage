(function(){
	function createButton(){
		const btn = document.createElement('button');
		btn.textContent = 'Narrate Image';
		btn.style.marginLeft = '8px';
		btn.style.padding = '6px 10px';
		btn.style.borderRadius = '6px';
		btn.style.border = '1px solid #ccc';
		btn.style.background = '#f8f8f8';
		btn.addEventListener('mouseenter',()=>btn.style.background='#f0f0f0');
		btn.addEventListener('mouseleave',()=>btn.style.background='#f8f8f8');
		return btn;
	}
	async function narrateDataUrl(dataUrl){
		const prefs = JSON.parse(localStorage.getItem('vsn_prefs')||'{}');
		const res = await fetch('/api/narrate',{
			method:'POST',
			headers:{'Content-Type':'application/json'},
			body: JSON.stringify({ image_base64: dataUrl, preferences: prefs })
		});
		const data = await res.json();
		if(!res.ok) throw new Error(data.error||'Failed');
		if (data.audio_base64){
			const a = new Audio(data.audio_base64);
			a.play().catch(()=>{});
		} else if (window.speechSynthesis){
			const utter = new SpeechSynthesisUtterance(data.text);
			utter.lang = (prefs && prefs.language) || 'en';
			window.speechSynthesis.speak(utter);
		}
	}
	function imgToDataUrl(img){
		const c = document.createElement('canvas');
		c.width = img.naturalWidth; c.height = img.naturalHeight;
		const ctx = c.getContext('2d');
		ctx.drawImage(img,0,0);
		return c.toDataURL('image/png');
	}
	function attach(){
		const imgs = document.querySelectorAll('img:not([data-vsn-attached])');
		imgs.forEach(img=>{
			img.setAttribute('data-vsn-attached','1');
			const btn = createButton();
			btn.addEventListener('click', async ()=>{
				try{
					const dataUrl = imgToDataUrl(img);
					await narrateDataUrl(dataUrl);
				}catch(e){ alert('Narration failed: '+e.message); }
			});
			img.insertAdjacentElement('afterend', btn);
		});
	}
	attach();
	const mo = new MutationObserver(attach);
	mo.observe(document.documentElement,{childList:true,subtree:true});
})();
