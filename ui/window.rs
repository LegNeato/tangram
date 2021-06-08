use crate as ui;
use pinwheel::prelude::*;

pub enum WindowShade {
	Code,
	Default,
}

#[derive(ComponentBuilder)]
pub struct Window {
	#[optional]
	pub padding: Option<bool>,
	#[children]
	pub children: Vec<Node>,
}

impl Component for Window {
	fn into_node(self) -> Node {
		let padding = self.padding.unwrap_or(true);
		let window_body_class = classes!(
			"window-body",
			if padding {
				Some("window-body-padding")
			} else {
				None
			},
		);
		div()
			.class("window-wrapper")
			.child(
				div()
					.class("window-topbar")
					.child(
						div()
							.class("window-topbar-button")
							.style(style::BACKGROUND_COLOR, ui::colors::RED),
					)
					.child(
						div()
							.class("window-topbar-button")
							.style(style::BACKGROUND_COLOR, ui::colors::YELLOW),
					)
					.child(
						div()
							.class("window-topbar-button")
							.style(style::BACKGROUND_COLOR, ui::colors::GREEN),
					),
			)
			.child(
				div()
					.attribute("class", window_body_class)
					.child(self.children),
			)
			.into_node()
	}
}
